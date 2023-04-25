from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.minibatch_sampler import init_sampler
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec

    from benchmark_utils.oracles import MultiLogRegOracle, DataCleaningOracle

    from benchopt.utils import profile

    import jax
    import jax.numpy as jnp
    from functools import partial


class Solver(BaseSolver):
    """Stochastic Bi-level Algorithm (SOBA)."""
    name = 'SOBA'

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
        'framework': [None, 'numba'],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def skip(self, f_train, f_val, **kwargs):
        if self.framework == 'numba':
            if self.batch_size == 'full':
                return True, "Numba is not useful for full bach resolution."
            elif isinstance(f_train(), MultiLogRegOracle):
                return True, "Numba implementation not available for " \
                      "Multiclass Logistic Regression."
            elif isinstance(f_val(), MultiLogRegOracle):
                return True, "Numba implementation not available for" \
                      "Multiclass Logistic Regression."
            elif isinstance(f_train(), DataCleaningOracle):
                return True, "Numba implementation not available for " \
                      "Datacleaning."
            elif isinstance(f_val(), DataCleaningOracle):
                return True, "Numba implementation not available for" \
                      "Datacleaning."
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
            self.soba = njit(soba)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )
        elif self.framework is None:
            self.soba = soba
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
            self.soba = partial(
                soba_jax,
                inner_sampler=inner_sampler,
                outer_sampler=outer_sampler
            )
        else:
            raise ValueError(f"Framework {self.framework} not supported.")

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        if self.framework == 'numba' or self.framework == 'jax':
            self.run_once(2)

    def run(self, callback):
        eval_freq = self.eval_freq
        rng = np.random.RandomState(self.random_state)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        if self.framework == "jax":
            v = jnp.zeros_like(inner_var)
            # Init lr scheduler
            step_sizes = jnp.array(
                [self.step_size, self.step_size / self.outer_ratio]
            )
            exponents = jnp.array(
                [.5, .5]
            )
            state_lr = dict(constants=step_sizes, exponents=exponents,
                            i_step=0)
            carry = dict(
                state_lr=state_lr,
                state_inner_sampler=self.state_inner_sampler,
                state_outer_sampler=self.state_outer_sampler,
            )
        else:
            v = np.zeros_like(inner_var)

            # Init lr scheduler
            step_sizes = np.array(
                [self.step_size, self.step_size / self.outer_ratio]
            )
            exponents = np.array(
                [.5, .5]
            )
            lr_scheduler = self.LearningRateScheduler(
                np.array(step_sizes, dtype=float), exponents
            )
            inner_sampler = self.MinibatchSampler(self.n_inner_samples,
                                                  self.batch_size_inner)
            outer_sampler = self.MinibatchSampler(self.n_outer_samples,
                                                  self.batch_size_outer)

        # Start algorithm
        while callback((inner_var, outer_var)):
            # Need to separate because numba does not support **kwargs
            if self.framework == 'jax':
                inner_var, outer_var, v, carry = self.soba(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v, max_iter=eval_freq, **carry
                )
            else:
                inner_var, outer_var, v = self.soba(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v,
                    inner_sampler=inner_sampler,
                    outer_sampler=outer_sampler,
                    lr_scheduler=lr_scheduler, max_iter=eval_freq,
                    seed=rng.randint(constants.MAX_SEED)
                )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


@profile
def soba(inner_oracle, outer_oracle, inner_var, outer_var, v,
         inner_sampler=None, outer_sampler=None, lr_scheduler=None, max_iter=1,
         seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_step_size, outer_step_size = lr_scheduler.get_lr()

        # Step.1 - get all gradients and compute the implicit gradient.
        slice_inner, _ = inner_sampler.get_batch()
        _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, slice_inner, inverse='id'
        )

        slice_outer, _ = outer_sampler.get_batch()
        grad_in_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        impl_grad -= cross_v

        # Step.2 - update inner variable with SGD.
        inner_var -= inner_step_size * grad_inner_var

        # Step.3 - update auxillary variable v with SGD
        v -= inner_step_size * (hvp - grad_in_outer)

        # Step.4 - update outer_variable with SGD
        outer_var -= outer_step_size * impl_grad

        # Use prox to make sure we do not diverge
        # inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

    return inner_var, outer_var, v


@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('inner_sampler', 'outer_sampler', 'max_iter'))
def soba_jax(f_inner, f_outer, inner_var, outer_var, v,
             state_inner_sampler=None, state_outer_sampler=None, state_lr=None,
             inner_sampler=None, outer_sampler=None, max_iter=1):
    def soba_one_iter(carry, _):
        grad_inner = jax.grad(f_inner, argnums=0)
        grad_outer = jax.grad(f_outer, argnums=(0, 1))

        inner_var, outer_var, v, state_lr, state_inner_sampler, \
            state_outer_sampler = carry

        (inner_step_size, outer_step_size), state_lr = update_lr(state_lr)

        # Step.1 - get all gradients and compute the implicit gradient.
        start_inner, state_inner_sampler = inner_sampler(**state_inner_sampler)
        grad_inner_var, vjp_train = jax.vjp(
            lambda z, x: grad_inner(z, x, start_inner), inner_var,
            outer_var
        )
        hvp, cross_v = vjp_train(v)

        start_outer, state_outer_sampler = outer_sampler(**state_outer_sampler)
        grad_in_outer, impl_grad = grad_outer(
            inner_var, outer_var, start_outer
        )
        impl_grad -= cross_v

        # Step.2 - update inner variable with SGD.
        inner_var -= inner_step_size * grad_inner_var[0]

        # Step.3 - update auxillary variable v with SGD
        v -= inner_step_size * (hvp - grad_in_outer)

        # Step.4 - update outer_variable with SGD
        outer_var -= outer_step_size * impl_grad

        # #Use prox to make sure we do not diverge
        # # inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        carry = inner_var, outer_var, v, state_lr, state_inner_sampler, \
            state_outer_sampler

        return carry, _

    init = (inner_var, outer_var, v, state_lr, state_inner_sampler,
            state_outer_sampler)
    carry, _ = jax.lax.scan(
        soba_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )
    return carry[0], carry[1], carry[2], dict(
        state_lr=carry[3],
        state_inner_sampler=carry[4],
        state_outer_sampler=carry[5]
    )
