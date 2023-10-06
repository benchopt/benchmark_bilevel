from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.get_memory import get_memory
    from benchmark_utils.minibatch_sampler import init_sampler
    from benchmark_utils.hessian_approximation import joint_shia
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.hessian_approximation import joint_shia_jax
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler
    from benchmark_utils.oracles import MultiLogRegOracle, DataCleaningOracle

    import jax
    import jax.numpy as jnp
    from functools import partial


class Solver(BaseSolver):
    """Momentum-based Recursive Bilevel Optimizer (MRBO).

    J. Yang, K. Ji, Y. Liang. "Provabily Faster Algorithms for Bilevel
    Optimization". NeurIPS 2021"""
    name = 'MRBO'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_shia_steps': [10],
        'batch_size': [64],
        'eta': [.5],
        'eval_freq': [128],
        'random_state': [1],
        'framework': ["jax"]
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

        if self.batch_size == "full":
            self.batch_size_inner = n_inner_samples
            self.batch_size_outer = n_outer_samples
        else:
            self.batch_size_inner = self.batch_size
            self.batch_size_outer = self.batch_size

        if self.framework == 'numba':
            # JIT necessary functions and classes
            njit_mrbo = njit(_mrbo)
            njit_joint_shia = njit(joint_shia)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

            def mrbo(*args, **kwargs):
                return njit_mrbo(njit_joint_shia, *args, **kwargs)
            self.mrbo = mrbo
        elif self.framework == "none":
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

            def mrbo(*args, **kwargs):
                return _mrbo(joint_shia, *args, **kwargs)
            self.mrbo = mrbo
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
            self.mrbo = partial(
                mrbo_jax,
                joint_shia=joint_shia_jax,
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
        eval_freq = self.eval_freq
        memory_start = get_memory()

        # Init variables
        inner_var = self.inner_var.copy()
        outer_var = self.outer_var.copy()
        if self.framework == 'jax':
            memory_inner = jnp.zeros((2, *inner_var.shape))
            memory_outer = jnp.zeros((2, *outer_var.shape))
            step_sizes = jnp.array(  # (inner_ss, hia_lr, eta, outer_ss)
                [
                    self.step_size,
                    self.step_size,
                    self.eta,
                    self.step_size / self.outer_ratio,
                ]
            )
            exponents = jnp.array([1/3, 0, 2/3, 1/3])
            state_lr = init_lr_scheduler(step_sizes, exponents)

            carry = dict(
                state_lr=state_lr,
                state_inner_sampler=self.state_inner_sampler,
                state_outer_sampler=self.state_outer_sampler,
            )
        else:
            rng = np.random.RandomState(self.random_state)
            memory_inner = np.zeros((2, *inner_var.shape), inner_var.dtype)
            memory_outer = np.zeros((2, *outer_var.shape), outer_var.dtype)

            inner_sampler = self.MinibatchSampler(
                self.f_inner.n_samples, batch_size=self.batch_size_inner
            )
            outer_sampler = self.MinibatchSampler(
                self.f_outer.n_samples, batch_size=self.batch_size_outer
            )
            step_sizes = np.array(  # (inner_ss, hia_lr, eta, outer_ss)
                [
                    self.step_size,
                    self.step_size,
                    self.eta,
                    self.step_size / self.outer_ratio,
                ]
            )
            exponents = np.array([1/3, 0, 2/3, 1/3])
            lr_scheduler = self.LearningRateScheduler(
                np.array(step_sizes, dtype=float), exponents
            )

        # Start algorithm
        while callback():
            if self.framework == 'jax':
                inner_var, outer_var, memory_inner, memory_outer, \
                    carry = self.mrbo(
                        self.f_inner, self.f_outer, inner_var, outer_var,
                        memory_inner, memory_outer,
                        n_shia_steps=self.n_shia_steps, max_iter=eval_freq,
                        **carry
                    )
            else:
                inner_var, outer_var, memory_inner, memory_outer = self.mrbo(
                    self.f_inner, self.f_outer, inner_var, outer_var,
                    memory_inner, memory_outer, inner_sampler, outer_sampler,
                    lr_scheduler, n_shia_steps=self.n_shia_steps,
                    max_iter=eval_freq, seed=rng.randint(constants.MAX_SEED)
                )
            memory_end = get_memory()
            self.inner_var = inner_var
            self.outer_var = outer_var
            self.memory = memory_end - memory_start
            self.memory /= 1e6

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var,
                    memory=self.memory)


def _mrbo(joint_shia, inner_oracle, outer_oracle, inner_var, outer_var,
          memory_inner, memory_outer, inner_sampler, outer_sampler,
          lr_scheduler, n_shia_steps=1, max_iter=1, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_lr, hia_lr, eta, outer_lr = lr_scheduler.get_lr()

        # Step.1 - Update direction for z with momentum
        slice_inner, _ = inner_sampler.get_batch()
        grad_inner_var = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice_inner
        )
        grad_inner_var_old = inner_oracle.grad_inner_var(
            memory_inner[0], memory_outer[0], slice_inner
        )
        memory_inner[1] = eta * grad_inner_var + (1-eta) * (
            memory_inner[1] + grad_inner_var - grad_inner_var_old
        )

        # Step.2 - Compute implicit grad approximation with HIA
        slice_outer, _ = outer_sampler.get_batch()
        grad_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        grad_outer_old, impl_grad_old = outer_oracle.grad(
            memory_inner[0], memory_outer[0], slice_outer
        )
        ihvp, ihvp_old = joint_shia(
            inner_oracle, inner_var, outer_var, grad_outer,
            memory_inner[0], memory_outer[0], grad_outer_old,
            hia_lr, sampler=inner_sampler, n_steps=n_shia_steps
        )
        impl_grad -= inner_oracle.cross(
            inner_var, outer_var, ihvp, slice_inner
        )
        impl_grad_old -= inner_oracle.cross(
            memory_inner[0], memory_outer[0], ihvp_old, slice_inner
        )

        # Step.3 - Update direction for x with momentum
        memory_outer[1] = eta * impl_grad + (1-eta) * (
            memory_outer[1] + impl_grad - impl_grad_old
        )

        # Step.4 - Save the current variables
        memory_inner[0] = inner_var
        memory_outer[0] = outer_var

        # Step.5 - update the variables with the directions
        inner_var -= inner_lr * memory_inner[1]
        outer_var -= outer_lr * memory_outer[1]

        # Step.6 - project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)
    return inner_var, outer_var, memory_inner, memory_outer


@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('joint_shia', 'n_shia_steps', 'inner_sampler',
                          'outer_sampler', 'max_iter'))
def mrbo_jax(f_inner, f_outer, inner_var, outer_var, memory_inner,
             memory_outer, state_inner_sampler=None, state_outer_sampler=None,
             state_lr=None, joint_shia=None, n_shia_steps=1,
             inner_sampler=None, outer_sampler=None, max_iter=1):
    grad_inner_fun = jax.grad(f_inner, argnums=0)
    grad_outer_fun = jax.grad(f_outer, argnums=(0, 1))

    def mrbo_one_iter(carry, _):

        (inner_lr, hia_lr, eta, outer_lr), carry['state_lr'] = update_lr(
            carry['state_lr']
        )

        # Step.1 - Update direction for z with momentum
        start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
            carry['state_inner_sampler']
        )
        grad_inner_var, vjp_fun = jax.vjp(
            lambda x: grad_inner_fun(carry['inner_var'], x, start_inner),
            carry['outer_var']
        )
        grad_inner_var_old, vjp_fun_old = jax.vjp(
            lambda x: grad_inner_fun(carry['memory_inner'][0], x, start_inner),
            carry['memory_outer'][0]
        )

        carry['memory_inner'] = carry['memory_inner'].at[1].set(
            grad_inner_var
            + (1-eta) * (carry['memory_inner'][1] - grad_inner_var_old)
        )

        # Step.2 - Compute implicit grad approximation with HIA
        start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
            carry['state_outer_sampler']
        )
        grad_outer, impl_grad = grad_outer_fun(
            carry['inner_var'], carry['outer_var'], start_outer
        )
        grad_outer_old, impl_grad_old = grad_outer_fun(
            carry['memory_inner'][0], carry['memory_outer'][0], start_outer
        )

        ihvp, ihvp_old, carry['state_inner_sampler'] = joint_shia(
            carry['inner_var'], carry['outer_var'], grad_outer,
            carry['memory_inner'][0], carry['memory_outer'][0], grad_outer_old,
            carry['state_inner_sampler'], hia_lr, sampler=inner_sampler,
            n_steps=n_shia_steps, grad_inner=grad_inner_fun
        )
        impl_grad -= vjp_fun(ihvp)[0]
        impl_grad_old -= vjp_fun_old(ihvp_old)[0]

        # Step.3 - Update direction for x with momentum
        carry['memory_outer'] = carry['memory_outer'].at[1].set(
            eta * impl_grad
            + (1-eta) * (carry['memory_outer'][1] + impl_grad - impl_grad_old)
        )

        # Step.4 - Save the current variables
        carry['memory_inner'] = carry['memory_inner'].at[0].set(
            carry['inner_var']
        )
        carry['memory_outer'] = carry['memory_outer'].at[0].set(
            carry['outer_var']
        )

        # Step.5 - update the variables with the directions
        carry['inner_var'] -= inner_lr * carry['memory_inner'][1]
        carry['outer_var'] -= outer_lr * carry['memory_outer'][1]

        return carry, _

    init = dict(
        inner_var=inner_var, outer_var=outer_var, memory_inner=memory_inner,
        memory_outer=memory_outer, state_lr=state_lr,
        state_inner_sampler=state_inner_sampler,
        state_outer_sampler=state_outer_sampler
    )
    carry, _ = jax.lax.scan(
        mrbo_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )
    return (
        carry['inner_var'], carry['outer_var'],
        carry['memory_inner'], carry['memory_outer'],
        {k: v for k, v in carry.items()
         if k not in ['inner_var', 'outer_var', 'memory_inner',
                      'memory_outer']}
    )
