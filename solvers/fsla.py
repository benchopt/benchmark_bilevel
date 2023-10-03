
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.minibatch_sampler import init_sampler
    from benchmark_utils.get_memory import get_memory
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler
    from benchmark_utils.oracles import MultiLogRegOracle, DataCleaningOracle

    import jax
    import jax.numpy as jnp
    from functools import partial


class Solver(BaseSolver):
    """Fully Single Loop Algorithm (FSLA).

    J. Li, B. Gu and H. Huang. "A Fully Single Loop Algorithm for Bilevel
    Optimization without Hessian Inverse". AAAI 2022"""
    name = 'FSLA'

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
        'framework': ["jax"],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def skip(self, f_train, f_val, **kwargs):
        if self.framework == 'numba':
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

        try:
            f_train(framework=self.framework)
        except NotImplementedError:
            return (
                True,
                f"Framework {self.framework} not compatible with "
                f"oracle {f_train()}"
            )
        return False, None

    def set_objective(self, f_train, f_val, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):
        self.f_inner = f_train(framework=self.framework)
        self.f_outer = f_val(framework=self.framework)
        self.n_inner_samples = n_inner_samples
        self.n_outer_samples = n_outer_samples

        # Init sampler and lr scheduler
        if self.batch_size == "full":
            self.batch_size_inner = n_inner_samples
            self.batch_size_outer = n_outer_samples
        else:
            self.batch_size_inner = self.batch_size
            self.batch_size_outer = self.batch_size

        if self.framework == 'numba':
            # JIT necessary functions and classes
            self.fsla = njit(fsla)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )
        elif self.framework == "none":
            self.fsla = fsla
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler
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
            self.fsla = partial(
                fsla_jax,
                inner_sampler=inner_sampler,
                outer_sampler=outer_sampler
            )
        else:
            raise ValueError(f"Framework {self.framework} not supported.")

        self.inner_var = inner_var0
        self.outer_var = outer_var0
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.memory = 0

    def warm_up(self):
        if self.framework in ['numba', 'jax']:
            self.run_once(2)
            self.inner_var = self.inner_var0
            self.outer_var = self.outer_var0

    def run(self, callback):
        eval_freq = self.eval_freq  # // self.batch_size

        memory_start = get_memory()
        memory_end = memory_start
        # Init variables
        inner_var = self.inner_var.copy()
        outer_var = self.outer_var.copy()
        if self.framework == 'jax':
            v = jnp.zeros_like(inner_var)
            memory_outer = jnp.zeros((2, *outer_var.shape))
            step_sizes = jnp.array(
                [self.step_size, self.step_size,
                 self.step_size / self.outer_ratio]
            )
            # Use 1 / sqrt(t) for the learning rates
            exponents = 0.5 * jnp.ones(len(step_sizes))
            state_lr = init_lr_scheduler(step_sizes, exponents)
            carry = dict(
                state_lr=state_lr,
                state_inner_sampler=self.state_inner_sampler,
                state_outer_sampler=self.state_outer_sampler,
            )
        else:
            rng = np.random.RandomState(self.random_state)
            v = np.zeros_like(inner_var)
            memory_outer = np.zeros((2, *outer_var.shape))

            inner_sampler = self.MinibatchSampler(
                self.f_inner.n_samples, batch_size=self.batch_size_inner
            )
            outer_sampler = self.MinibatchSampler(
                self.f_outer.n_samples, batch_size=self.batch_size_outer
            )
            step_sizes = np.array(
                [self.step_size, self.step_size,
                 self.step_size / self.outer_ratio]
            )
            # Use 1 / sqrt(t) for the learning rates
            exponents = 0.5 * np.ones(len(step_sizes))
            lr_scheduler = self.LearningRateScheduler(
                np.array(step_sizes, dtype=float), exponents
            )

        # Start algorithm
        while callback():
            if self.framework == 'jax':
                inner_var, outer_var, v, memory_outer, carry = self.fsla(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v, memory_outer,
                    max_iter=eval_freq, **carry
                )
            else:
                inner_var, outer_var, v, memory_outer = self.fsla(
                    self.f_inner, self.f_outer, inner_var, outer_var, v,
                    memory_outer,
                    inner_sampler=inner_sampler,
                    outer_sampler=outer_sampler,
                    lr_scheduler=lr_scheduler, max_iter=eval_freq,
                    seed=rng.randint(constants.MAX_SEED)
                )
            memory_end = get_memory()
            self.inner_var = inner_var
            self.outer_var = outer_var
            self.memory = memory_end - memory_start
            self.memory /= 1e6

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var,
                    memory=self.memory)


def fsla(inner_oracle, outer_oracle, inner_var, outer_var, v, memory_outer,
         inner_sampler=None, outer_sampler=None, lr_scheduler=None, max_iter=1,
         seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_lr, eta, outer_lr = lr_scheduler.get_lr()

        # Step.1 - SGD step on the inner problem
        slice_inner, _ = inner_sampler.get_batch()
        grad_inner_var = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice_inner
        )
        inner_var_old = inner_var.copy()
        inner_var -= inner_lr * grad_inner_var

        # Step.2 - SGD step on the auxillary variable v
        slice_inner2, _ = inner_sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, slice_inner2)
        slice_outer, _ = outer_sampler.get_batch()
        grad_in_outer = outer_oracle.grad_inner_var(
            inner_var, outer_var, slice_outer
        )
        v_old = v.copy()
        v -= inner_lr * (hvp - grad_in_outer)

        # Step.3 - compute the implicit gradient estimates, for the old
        # and new variables
        slice_outer2, _ = outer_sampler.get_batch()
        impl_grad = outer_oracle.grad_outer_var(
            inner_var, outer_var, slice_outer2
        )
        impl_grad_old = outer_oracle.grad_outer_var(
            inner_var_old, memory_outer[0], slice_outer2
        )
        slice_inner3, _ = inner_sampler.get_batch()
        impl_grad -= inner_oracle.cross(inner_var, outer_var, v, slice_inner3)
        impl_grad_old -= inner_oracle.cross(
            inner_var_old, memory_outer[0], v_old, slice_inner3
        )

        # Step.4 - update direction with momentum
        memory_outer[1] = (
            impl_grad + (1-eta) * (memory_outer[1] - impl_grad_old)
        )

        # Step.5 - update the outer variable
        memory_outer[0] = outer_var
        outer_var -= outer_lr * memory_outer[1]

        # Step.6 - project back to the constraint set
        # inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

    return inner_var, outer_var, v, memory_outer


@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('inner_sampler', 'outer_sampler', 'max_iter'))
def fsla_jax(f_inner, f_outer, inner_var, outer_var, v, memory_outer,
             state_inner_sampler=None, state_outer_sampler=None, state_lr=None,
             inner_sampler=None, outer_sampler=None, max_iter=1):
    grad_inner_fun = jax.grad(f_inner, argnums=0)
    grad_outer_fun = jax.grad(f_outer, argnums=(0, 1))

    def fsla_one_iter(carry, _):

        (inner_lr, eta, outer_lr), carry['state_lr'] = update_lr(
            carry['state_lr']
        )

        # Step.1 - SGD step on the inner problem
        start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
            carry['state_inner_sampler']
        )
        grad_inner_var = grad_inner_fun(carry['inner_var'], carry['outer_var'],
                                        start_inner)
        inner_var_old = carry['inner_var'].copy()
        carry['inner_var'] -= inner_lr * grad_inner_var

        # Step.2 - SGD step on the auxillary variable v
        start_inner2, *_, carry['state_inner_sampler'] = inner_sampler(
            carry['state_inner_sampler']
        )
        _, hvp_fun = jax.vjp(
            lambda z: grad_inner_fun(z, carry['outer_var'], start_inner2),
            carry['inner_var']
        )

        start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
            carry['state_outer_sampler']
        )
        grad_outer_in, _ = grad_outer_fun(carry['inner_var'],
                                          carry['outer_var'],
                                          start_outer)
        v_old = carry['v'].copy()
        carry['v'] -= inner_lr * (hvp_fun(carry['v'])[0] - grad_outer_in)

        # Step.3 - compute the implicit gradient estimates, for the old
        # and new variables
        start_outer2, *_, carry['state_outer_sampler'] = outer_sampler(
            carry['state_outer_sampler']
        )
        _, impl_grad = grad_outer_fun(
            carry['inner_var'], carry['outer_var'], start_outer2
        )
        _, impl_grad_old = grad_outer_fun(
            inner_var_old, carry['memory_outer'][0], start_outer2
        )
        start_inner3, *_, carry['state_inner_sampler'] = inner_sampler(
            carry['state_inner_sampler']
        )
        _, cross_v_fun = jax.vjp(
            lambda x: grad_inner_fun(carry['inner_var'], x, start_inner3),
            carry['outer_var']
        )
        _, cross_v_fun_old = jax.vjp(
            lambda x: grad_inner_fun(inner_var_old, x, start_inner3),
            carry['memory_outer'][0]
        )
        impl_grad -= cross_v_fun(carry['v'])[0]
        impl_grad_old -= cross_v_fun_old(v_old)[0]

        # Step.4 - update direction with momentum
        carry['memory_outer'] = carry['memory_outer'].at[1].set(
            impl_grad + (1-eta) * (carry['memory_outer'][1] - impl_grad_old)
        )

        # Step.5 - update the outer variable
        carry['memory_outer'] = carry['memory_outer'].at[0].set(
            carry['outer_var']
        )
        carry['outer_var'] -= outer_lr * carry['memory_outer'][1]
        return carry, _

    init = dict(
        inner_var=inner_var, outer_var=outer_var, v=v,
        memory_outer=memory_outer, state_lr=state_lr,
        state_inner_sampler=state_inner_sampler,
        state_outer_sampler=state_outer_sampler
    )
    carry, _ = jax.lax.scan(
        fsla_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )
    return (
        carry['inner_var'], carry['outer_var'], carry['v'],
        carry['memory_outer'],
        {k: v for k, v in carry.items()
         if k not in ['inner_var', 'outer_var', 'v', 'memory_outer']}
    )
