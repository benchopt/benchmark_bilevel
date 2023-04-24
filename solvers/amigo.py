
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.sgd_inner import sgd_inner
    from benchmark_utils.hessian_approximation import sgd_v
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec

    from benchmark_utils.oracles import MultiLogRegOracle, DataCleaningOracle


class Solver(BaseSolver):
    """Two loops solver."""
    name = 'AmIGO'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'eval_freq': [128],
        'n_inner_step': [10],
        'batch_size': [64],
        'random_state': [1],
        'framework': [None, 'numba']
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

    def set_objective(self, f_train, f_val, inner_var0, outer_var0):
        self.f_inner = f_train(framework=self.framework)
        self.f_outer = f_val(framework=self.framework)

        if self.framework == 'numba':
            # JIT necessary functions and classes
            self.sgd_v = njit(sgd_v)
            njit_amigo = njit(_amigo)
            self.sgd_inner = njit(sgd_inner)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

            def amigo(*args, seed=None):
                return njit_amigo(self.sgd_inner, self.sgd_v, *args, seed=seed)
            self.amigo = amigo
        elif self.framework is None:
            self.sgd_v = sgd_v
            self.sgd_inner = sgd_inner
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

            def amigo(*args, seed=None):
                return _amigo(self.sgd_inner, self.sgd_v, *args, seed=seed)
            self.amigo = amigo
        elif self.framework == 'jax':
            raise NotImplementedError("Jax version not implemented yet")
        else:
            raise ValueError(f"Framework {self.framework} not supported.")

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        if self.framework == 'numba':
            self.run_once(2)

    def run(self, callback):
        eval_freq = self.eval_freq
        rng = np.random.RandomState(self.random_state)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        v = np.zeros_like(inner_var)

        # Init sampler and lr scheduler
        if self.batch_size == 'full':
            batch_size_inner = self.f_inner.n_samples
            batch_size_outer = self.f_outer.n_samples
        else:
            batch_size_inner = self.batch_size
            batch_size_outer = self.batch_size
        inner_sampler = self.MinibatchSampler(
            self.f_inner.n_samples, batch_size=batch_size_inner
        )
        outer_sampler = self.MinibatchSampler(
            self.f_outer.n_samples, batch_size=batch_size_outer
        )
        step_sizes = np.array(
            [self.step_size, self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = np.zeros(3)
        lr_scheduler = self.LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        # Start algorithm
        inner_var = self.sgd_inner(
            self.f_inner, inner_var, outer_var, self.step_size,
            inner_sampler=inner_sampler, n_inner_step=self.n_inner_step,
        )
        while callback((inner_var, outer_var)):
            inner_var, outer_var, v = self.amigo(
                self.f_inner, self.f_outer,
                inner_var, outer_var, v, inner_sampler, outer_sampler,
                eval_freq, self.n_inner_step, 10, lr_scheduler,
                seed=rng.randint(constants.MAX_SEED)
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


def _amigo(sgd_inner, sgd_v, inner_oracle, outer_oracle, inner_var, outer_var,
           v, inner_sampler, outer_sampler, max_iter, n_inner_step, n_v_step,
           lr_scheduler, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_step_size, v_step_size, outer_step_size = lr_scheduler.get_lr()

        # Get outer gradient
        outer_slice, _ = outer_sampler.get_batch()
        grad_in, grad_out = outer_oracle.grad(
            inner_var, outer_var, outer_slice
        )

        # compute SGD for the auxillary variable
        v = sgd_v(inner_oracle, inner_var, outer_var, v, grad_in,
                  inner_sampler, n_v_step, v_step_size)

        inner_slice, _ = inner_sampler.get_batch()
        cross_hvp = inner_oracle.cross(inner_var, outer_var, v, inner_slice)
        implicit_grad = grad_out - cross_hvp

        outer_var -= outer_step_size * implicit_grad
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        inner_var = sgd_inner(
            inner_oracle, inner_var, outer_var, inner_step_size,
            inner_sampler=inner_sampler, n_inner_step=n_inner_step
        )
    return inner_var, outer_var, v
