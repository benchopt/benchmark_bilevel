
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit, int64, float64
    from numba.experimental import jitclass
    constants = import_ctx.import_from('constants')
    sgd_inner = import_ctx.import_from('sgd_inner', 'sgd_inner')
    sgd_v = import_ctx.import_from('hessian_approximation', 'sgd_v')
    MinibatchSampler = import_ctx.import_from(
        'minibatch_sampler', 'MinibatchSampler'
    )
    LearningRateScheduler = import_ctx.import_from(
        'learning_rate_scheduler', 'LearningRateScheduler'
    )


class Solver(BaseSolver):
    """Two loops solver."""
    name = 'AmIGO'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': constants.STEP_SIZES,
        'outer_ratio': constants.OUTER_RATIOS,
        'n_inner_step': constants.N_INNER_STEPS,
        'batch_size': constants.BATCH_SIZES,
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_train, f_test, inner_var0, outer_var0, numba):
        if numba:
            self.f_inner = f_train.numba_oracle
            self.f_outer = f_test.numba_oracle
            self.sgd_inner = njit(sgd_inner)
            self.sgd_v = njit(sgd_v)

            spec_minibatch_sampler = [
                ('n_samples', int64),
                ('batch_size', int64),
                ('i_batch', int64),
                ('n_batches', int64),
                ('batch_order', int64[:]),
            ]
            self.MinibatchSampler = jitclass(MinibatchSampler,
                                             spec_minibatch_sampler)

            spec_scheduler = [
                ('i_step', int64),
                ('constants', float64[:]),
                ('exponents', float64[:])
            ]
            self.LearningRateScheduler = jitclass(LearningRateScheduler,
                                                  spec_scheduler)

            def amigo(sgd_inner, sgd_v):
                def f(*args, seed=None):
                    return njit(_amigo)(sgd_inner, sgd_v, *args, seed=seed)
                return f
            self.amigo = amigo(self.sgd_inner, self.sgd_v)
        else:
            self.f_inner = f_train
            self.f_outer = f_test
            self.sgd_inner = sgd_inner
            self.sgd_v = sgd_v
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

            def amigo(sgd_inner, sgd_v):
                def f(*args, seed=None):
                    return _amigo(sgd_inner, sgd_v, *args, seed=seed)
                return f
            self.amigo = amigo(self.sgd_inner, self.sgd_v)

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.numba = numba

    def run(self, callback):
        eval_freq = constants.EVAL_FREQ
        rng = np.random.RandomState(constants.RANDOM_STATE)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        v = self.f_outer.grad_inner_var(inner_var, outer_var, np.array([0]))

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
