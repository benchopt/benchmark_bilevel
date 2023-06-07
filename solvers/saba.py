from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit, prange
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.minibatch_sampler import init_sampler
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec
    from benchmark_utils.oracles import MultiLogRegOracle, DataCleaningOracle
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler

    import jax
    import jax.numpy as jnp
    from functools import partial

    # from benchopt.utils import profile


class Solver(BaseSolver):
    """Stochastic Average Bi-level Algorithm."""
    name = 'SABA'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'batch_size': [64],
        'eval_freq': [128],
        'random_state': [1],
        'framework': ["none"],
        'init_memory': ["zero"],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def skip(self, f_train, f_val, **kwargs):
        if self.framework == 'Numba':
            if self.batch_size == 'full':
                return True, "Numba is not useful for full bach resolution."
            elif isinstance(f_train(),
                            (MultiLogRegOracle, DataCleaningOracle)):
                return True, "Numba implementation not available for " \
                      "this oracle."
            elif isinstance(f_val(), (MultiLogRegOracle, DataCleaningOracle)):
                return True, "Numba implementation not available for" \
                      "this oracle."
        elif self.framework not in ['jax', 'none', 'numba']:
            return True, f"Framework {self.framework} not supported."
        return False, None

    def set_objective(self, f_train, f_val, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):
        self.f_inner = f_train(framework=self.framework)
        self.f_outer = f_val(framework=self.framework)
        self.n_inner_samples = n_inner_samples
        self.n_outer_samples = n_outer_samples
        self.inner_size = f_train().variables_shape[0, 0]
        self.outer_size = f_train().variables_shape[1, 0]

        if self.batch_size == "full":
            self.batch_size_inner = n_inner_samples
            self.batch_size_outer = n_outer_samples
        else:
            self.batch_size_inner = self.batch_size
            self.batch_size_outer = self.batch_size

        if self.framework == 'numba':
            # JIT necessary functions and classes
            njit_saba = njit(_saba)
            njit_vr = njit(variance_reduction)
            njit_init_mem = njit(_init_memory)
            njit_init_mem_fb = njit(_init_memory_fb)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

            def init_memory(*args, **kwargs):
                return njit_init_mem(njit_init_mem_fb, *args, **kwargs)
            self.init_memory = init_memory

            def saba(*args, **kwargs):
                return njit_saba(njit_vr, *args, **kwargs)
            self.saba = saba
        elif self.framework == "none":
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

            def init_memory(*args, **kwargs):
                return _init_memory(_init_memory_fb, *args, **kwargs)
            self.init_memory = init_memory

            def saba(*args, **kwargs):
                return _saba(variance_reduction, *args, **kwargs)
            self.saba = saba
        elif self.framework == 'jax':
            self.f_inner = jax.jit(
                partial(self.f_inner, batch_size=self.batch_size_inner)
            )
            self.f_outer = jax.jit(
                partial(self.f_outer, batch_size=self.batch_size_outer)
            )
            inner_sampler, self.state_inner_sampler \
                = init_sampler(n_samples=n_inner_samples,
                               batch_size=self.batch_size_inner)
            outer_sampler, self.state_outer_sampler \
                = init_sampler(n_samples=n_outer_samples,
                               batch_size=self.batch_size_outer)
            self.weight_inner = self.batch_size_inner / n_inner_samples
            self.weight_outer = self.batch_size_outer / n_outer_samples

            def init_memory(*args, **kwargs):
                return _init_memory_jax(_init_memory_fb_jax, *args, **kwargs)
            self.init_memory = init_memory
            self.saba = partial(
                saba_jax,
                inner_sampler=inner_sampler,
                outer_sampler=outer_sampler,
                # variance_reduction=jax.jit(variance_reduction_jax)
            )
        else:
            raise ValueError(f"Framework {self.framework} not supported.")

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        if self.framework == 'numba' or self.framework == 'jax':
            self.run_once(2)

    def run(self, callback):
        eval_freq = self.eval_freq  # // self.batch_size
        rng = np.random.RandomState(self.random_state)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        if self.framework == 'jax':
            v = jnp.zeros_like(inner_var)

            step_sizes = jnp.array(
                [self.step_size, self.step_size / self.outer_ratio]
            )
            exponents = jnp.zeros(2)
            state_lr = dict(constants=step_sizes, exponents=exponents,
                            i_step=0)

            memory_inner_grad, memory_hvp, memory_cross_v, \
                memory_grad_in_outer, memory_grad_out_outer = self.init_memory(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v,
                    n_inner_samples=self.n_inner_samples,
                    n_outer_samples=self.n_outer_samples,
                    batch_size_inner=self.batch_size_inner,
                    batch_size_outer=self.batch_size_outer,
                    inner_size=self.inner_size,
                    outer_size=self.outer_size,
                )
            carry = dict(
                state_inner_sampler=self.state_inner_sampler,
                state_outer_sampler=self.state_outer_sampler,
                state_lr=state_lr,
                weight_inner=self.weight_inner,
                weight_outer=self.weight_outer,
            )
        else:
            v = np.zeros_like(inner_var)

            inner_sampler = self.MinibatchSampler(
                self.f_inner.n_samples, batch_size=self.batch_size_inner
            )
            outer_sampler = self.MinibatchSampler(
                self.f_outer.n_samples, batch_size=self.batch_size_outer
            )
            step_sizes = np.array(
                [self.step_size, self.step_size / self.outer_ratio]
            )
            exponents = np.zeros(2)
            lr_scheduler = self.LearningRateScheduler(
                np.array(step_sizes, dtype=float), exponents
            )

            memory_inner_grad, memory_hvp, memory_cross_v, \
                memory_grad_in_outer, memory_grad_out_outer = self.init_memory(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v, inner_sampler, outer_sampler
                )

        # Start algorithm
        while callback((inner_var, outer_var)):
            if self.framework == 'jax':
                inner_var, outer_var, v, memory_inner_grad, memory_hvp,\
                    memory_cross_v, memory_grad_in_outer,\
                    memory_grad_out_outer, carry = self.saba(
                        self.f_inner, self.f_outer, inner_var, outer_var,
                        v, memory_inner_grad, memory_hvp, memory_cross_v,
                        memory_grad_in_outer, memory_grad_out_outer,
                        max_iter=eval_freq, **carry
                    )
            else:
                inner_var, outer_var, v = self.saba(
                    self.f_inner, self.f_outer, inner_var, outer_var, v,
                    memory_inner_grad, memory_hvp, memory_cross_v,
                    memory_grad_in_outer, memory_grad_out_outer,
                    inner_sampler=inner_sampler, outer_sampler=outer_sampler,
                    lr_scheduler=lr_scheduler, max_iter=eval_freq,
                    seed=rng.randint(constants.MAX_SEED)
                )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


def _init_memory(
    _init_memory_fb,
    inner_oracle,
    outer_oracle,
    inner_var,
    outer_var,
    v,
    inner_sampler,
    outer_sampler,
    mode="zero",
):
    if mode == "full":
        memories = _init_memory_fb(
            inner_oracle,
            outer_oracle,
            inner_var,
            outer_var,
            v,
            inner_sampler,
            outer_sampler,
        )
        for mem in memories:
            mem[-1] = mem[:-1].sum(axis=0) / mem[:-1].shape[0]
    else:
        n_outer = outer_sampler.n_batches
        n_inner = inner_sampler.n_batches
        inner_size, outer_size = inner_oracle.variables_shape
        memories = (
            np.zeros((n_inner + 1, inner_size[0])),
            np.zeros((n_inner + 1, inner_size[0])),
            np.zeros((n_inner + 1, outer_size[0])),
            np.zeros((n_outer + 1, inner_size[0])),
            np.zeros((n_outer + 1, outer_size[0])),
        )
    return memories


def _init_memory_fb(inner_oracle, outer_oracle, inner_var, outer_var, v,
                    inner_sampler, outer_sampler):
    n_outer = outer_sampler.n_batches
    n_inner = inner_sampler.n_batches
    inner_var_shape, outer_var_shape = inner_oracle.variables_shape.ravel()
    memory_inner_grad = np.zeros((n_inner + 1, inner_var_shape))
    memory_hvp = np.zeros((n_inner + 1, inner_var_shape))
    memory_cross_v = np.zeros((n_inner + 1, outer_var_shape))
    for _ in prange(n_inner):
        slice_inner, (id_inner, weight) = inner_sampler.get_batch()
        _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, slice_inner, inverse='id'
        )
        memory_inner_grad[id_inner, :] = grad_inner_var
        memory_inner_grad[-1, :] += weight * grad_inner_var
        memory_hvp[id_inner, :] = hvp
        memory_hvp[-1, :] += weight * hvp
        memory_cross_v[id_inner, :] = cross_v
        memory_cross_v[-1, :] += weight * cross_v

    memory_outer_grad_in = np.zeros((n_outer + 1, inner_var_shape))
    memory_outer_grad_out = np.zeros((n_outer + 1, outer_var_shape))
    for id_outer in prange(n_outer):
        slice_outer, (id_outer, weight) = outer_sampler.get_batch()
        grad_in, grad_out = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        memory_outer_grad_in[id_outer, :] = grad_in
        memory_outer_grad_in[-1, :] += weight * memory_outer_grad_in[id_outer]
        memory_outer_grad_out[id_outer, :] = grad_out
        memory_outer_grad_out[-1, :] += weight*memory_outer_grad_out[id_outer]

    return memory_inner_grad, memory_hvp, memory_cross_v, \
        memory_outer_grad_in, memory_outer_grad_in


def _init_memory_jax(
    _init_memory_fb,
    inner_oracle,
    outer_oracle,
    inner_var,
    outer_var,
    v,
    n_inner_samples=1,
    n_outer_samples=1,
    batch_size_inner=1,
    batch_size_outer=1,
    inner_size=1,
    outer_size=1,
    mode="zero",
):
    if mode == "full":
        grad_inner = jax.grad(inner_oracle, argnums=0)
        grad_outer = jax.grad(outer_oracle, argnums=(0, 1))
        memories = _init_memory_fb(
            grad_inner,
            grad_outer,
            inner_var,
            outer_var,
            v,
            n_inner_samples=n_inner_samples,
            n_outer_samples=n_outer_samples,
            batch_size_inner=batch_size_inner,
            batch_size_outer=batch_size_outer,
            inner_size=inner_size,
            outer_size=outer_size,
        )
        for mem in memories:
            mem = mem.at[-1].set(mem[:-1].sum(axis=0) / mem[:-1].shape[0])
    else:
        n_outer = (n_outer_samples + batch_size_outer - 1) // batch_size_outer
        n_inner = (n_inner_samples + batch_size_inner - 1) // batch_size_inner
        memories = (
            jnp.zeros((n_inner + 1, inner_size)),
            jnp.zeros((n_inner + 1, inner_size)),
            jnp.zeros((n_inner + 1, outer_size)),
            jnp.zeros((n_outer + 1, inner_size)),
            jnp.zeros((n_outer + 1, outer_size)),
        )
    return memories


def _init_memory_fb_jax(
        grad_inner,
        grad_outer,
        inner_var,
        outer_var,
        v,
        inner_sampler,
        outer_sampler,
        state_inner_sampler,
        state_outer_sampler,
        n_inner_samples=1,
        n_outer_samples=1,
        batch_size_inner=1,
        batch_size_outer=1,
        inner_size=1,
        outer_size=1,
):
    n_outer = (n_outer_samples + batch_size_outer - 1) // batch_size_outer
    n_inner = (n_inner_samples + batch_size_inner - 1) // batch_size_inner
    memory_inner_grad = jnp.zeros((n_inner + 1, inner_size))
    memory_hvp = jnp.zeros((n_inner + 1, inner_size))
    memory_cross_v = jnp.zeros((n_inner + 1, outer_size))
    for _ in range(n_inner):
        start_inner, state_inner_sampler = inner_sampler(**state_inner_sampler)
        id_inner = state_inner_sampler['batch_order'][
            state_inner_sampler['i_batch']
        ]
        weight = batch_size_inner / n_inner_samples
        grad_inner_var, vjp_train = jax.vjp(
                    lambda z, x: grad_inner(z, x, start_inner), inner_var,
                    outer_var
                )
        hvp, cross_v = vjp_train(v)

        memory_inner_grad = memory_inner_grad.at[id_inner].set(grad_inner_var)
        memory_inner_grad = memory_inner_grad.at[-1].add(weight
                                                         * grad_inner_var)
        memory_hvp = memory_hvp.at[id_inner].set(hvp)
        memory_hvp = memory_hvp.at[-1].add(weight * hvp)
        memory_cross_v = memory_cross_v.at[id_inner].set(cross_v)
        memory_cross_v = memory_cross_v.at[-1].add(weight * cross_v)

    memory_outer_grad_in = np.zeros((n_outer + 1, inner_size))
    memory_outer_grad_out = np.zeros((n_outer + 1, outer_size))
    for id_outer in prange(n_outer):
        start_outer, state_outer_sampler = outer_sampler(**state_outer_sampler)
        id_outer = state_outer_sampler['batch_order'][
            state_outer_sampler['i_batch']
        ]
        weight = batch_size_outer / n_outer_samples

        grad_in, grad_out = grad_outer(inner_var, outer_var, start_outer)

        memory_outer_grad_in = memory_outer_grad_in.at[id_outer].set(grad_in)
        memory_outer_grad_in = memory_outer_grad_in.at[-1].add(weight
                                                               * grad_in)
        memory_outer_grad_out = memory_outer_grad_out.at[id_outer].set(
            grad_out
        )
        memory_outer_grad_out = memory_outer_grad_out.at[-1].add(weight
                                                                 * grad_out)

    return memory_inner_grad, memory_hvp, memory_cross_v, \
        memory_outer_grad_in, memory_outer_grad_out


def variance_reduction(grad, memory, vr_info):
    idx, weigth = vr_info
    diff = grad - memory[idx]
    direction = diff + memory[-1]
    memory[-1] += diff * weigth
    memory[idx, :] = grad
    return direction


def _saba(variance_reduction, inner_oracle, outer_oracle, inner_var, outer_var,
          v, memory_inner_grad, memory_hvp, memory_cross_v,
          memory_grad_in_outer, memory_grad_out_outer, inner_sampler=None,
          outer_sampler=None, lr_scheduler=None, max_iter=1, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_step_size, outer_step_size = lr_scheduler.get_lr()

        # Get all gradient for the batch
        slice_inner, vr_inner = inner_sampler.get_batch()
        _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, slice_inner, inverse='id'
        )
        slice_outer, vr_outer = outer_sampler.get_batch()
        grad_in_outer, grad_out_outer = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        # here memory_*[-1] corresponds to the running average of
        # the gradients
        grad_inner_var = variance_reduction(
            grad_inner_var, memory_inner_grad, vr_inner
        )
        hvp = variance_reduction(hvp, memory_hvp, vr_inner)
        grad_in_outer = variance_reduction(
            grad_in_outer, memory_grad_in_outer, vr_outer
        )

        cross_v = variance_reduction(
            cross_v, memory_cross_v, vr_inner
        )
        grad_out_outer = variance_reduction(grad_out_outer,
                                            memory_grad_out_outer, vr_outer)

        # Update the variables
        inner_var -= inner_step_size * grad_inner_var
        v -= inner_step_size * (hvp + grad_in_outer)
        outer_var -= outer_step_size * (cross_v + grad_out_outer)
    return inner_var, outer_var, v


@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('inner_sampler',
                          'outer_sampler', 'max_iter'))
def saba_jax(f_inner, f_outer, inner_var, outer_var, v, memory_inner_grad,
             memory_hvp, memory_cross_v, memory_grad_in_outer,
             memory_grad_out_outer,
             weight_inner=.1, weight_outer=.1,
             state_inner_sampler=None, state_outer_sampler=None, state_lr=None,
             inner_sampler=None, outer_sampler=None, max_iter=1):
    def variance_reduction(grad, memory, idx, weigth):
        diff = grad - memory.at[idx].get()
        direction = diff + memory.at[-1].get()
        memory = memory.at[-1].add(weigth * diff).at[idx].set(grad)
        return direction, memory

    def saba_one_iter(carry, _):
        grad_inner = jax.grad(f_inner, argnums=0)
        grad_outer = jax.grad(f_outer, argnums=(0, 1))
        (inner_step_size, outer_step_size), carry['state_lr'] = update_lr(
            carry['state_lr']
        )

        # Get all gradient for the batch
        start_inner, carry['state_inner_sampler'] = inner_sampler(
            **carry['state_inner_sampler']
        )
        grad_inner_var, vjp_train = jax.vjp(
            lambda z, x: grad_inner(z, x, start_inner), carry['inner_var'],
            carry['outer_var']
        )
        hvp, cross_v = vjp_train(carry['v'])

        start_outer, carry['state_outer_sampler'] = outer_sampler(
            **carry['state_outer_sampler']
        )
        grad_in_outer, grad_out_outer = grad_outer(carry['inner_var'],
                                                   carry['outer_var'],
                                                   start_outer)
        # here memory_*[-1] corresponds to the running average of
        # the gradients
        id_inner = carry['state_inner_sampler']['batch_order'][
            carry['state_inner_sampler']['i_batch']
        ]
        id_outer = carry['state_outer_sampler']['batch_order'][
            carry['state_outer_sampler']['i_batch']
        ]
        grad_inner_var, carry['memory_inner_grad'] = variance_reduction(
            grad_inner_var, carry['memory_inner_grad'], id_inner, weight_inner
        )
        hvp, carry['memory_hvp'] = variance_reduction(
            hvp, carry['memory_hvp'], id_inner, weight_inner
        )
        cross_v, carry['memory_cross_v'] = variance_reduction(
            cross_v, carry['memory_cross_v'], id_inner, weight_inner
        )

        grad_in_outer, carry['memory_grad_in_outer'] = variance_reduction(
            grad_in_outer, carry['memory_grad_in_outer'], id_outer,
            weight_outer
        )
        grad_out_outer, carry['memory_grad_out_outer'] = variance_reduction(
            grad_out_outer, carry['memory_grad_out_outer'], id_outer,
            weight_outer
        )

        # Update the variables
        carry['inner_var'] -= inner_step_size * grad_inner_var
        carry['v'] -= inner_step_size * (hvp + grad_in_outer)
        carry['outer_var'] -= outer_step_size * (cross_v + grad_out_outer)

        # #Use prox to make sure we do not diverge
        return carry, _

    init = dict(
        inner_var=inner_var, outer_var=outer_var, v=v,
        memory_inner_grad=memory_inner_grad, memory_hvp=memory_hvp,
        memory_cross_v=memory_cross_v,
        memory_grad_in_outer=memory_grad_in_outer,
        memory_grad_out_outer=memory_grad_out_outer,
        state_lr=state_lr, state_inner_sampler=state_inner_sampler,
        state_outer_sampler=state_outer_sampler,
        weight_inner=weight_inner, weight_outer=weight_outer,
    )
    carry, _ = jax.lax.scan(
        saba_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )
    return carry['inner_var'], carry['outer_var'], carry['v'], \
        carry['memory_inner_grad'], carry['memory_hvp'], \
        carry['memory_cross_v'], carry['memory_grad_in_outer'], \
        carry['memory_grad_out_outer'], \
        {k: v for k, v in carry.items()
         if k not in ['inner_var', 'outer_var', 'v', 'memory_inner_grad',
                      'memory_hvp', 'memory_cross_v', 'memory_grad_in_outer',
                      'memory_grad_out_outer']},
