
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
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
        'step_size': [.1],
        'outer_ratio': [.01],
        'n_inner_step': [10],
        'batch_size': constants.BATCH_SIZES,
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_train, f_test, inner_var0, outer_var0):
        self.f_inner = f_train
        self.f_outer = f_test
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        if self.batch_size == 'all':
            self.inner_batch_size = self.f_inner.n_samples
            self.outer_batch_size = self.f_outer.n_samples
        else:
            self.inner_batch_size = self.batch_size
            self.outer_batch_size = self.batch_size

    def run(self, callback):
        eval_freq = constants.EVAL_FREQ
        rng = np.random.RandomState(constants.RANDOM_STATE)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        v = self.f_outer.grad_inner_var(inner_var, outer_var, np.array([0]))

        # Init sampler and lr scheduler
        inner_sampler = MinibatchSampler(
            self.f_inner.n_samples, batch_size=self.batch_size
        )
        outer_sampler = MinibatchSampler(
            self.f_outer.n_samples, batch_size=self.batch_size
        )
        step_sizes = np.array(
            [self.step_size, self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = np.zeros(3)
        lr_scheduler = LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        # Start algorithm
        inner_var = sgd_inner(
            self.f_inner.numba_oracle, inner_var, outer_var, self.step_size,
            inner_sampler=inner_sampler, n_inner_step=self.n_inner_step,
        )
        while callback((inner_var, outer_var)):
            inner_var, outer_var, v = amigo(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, v, inner_sampler, outer_sampler,
                eval_freq, self.n_inner_step, self.n_inner_step, lr_scheduler,
                seed=rng.randint(constants.MAX_SEED)
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta

    def line_search(self, outer_var, grad):
        pass


# @njit
def amigo(inner_oracle, outer_oracle, inner_var, outer_var, v,
          inner_sampler, outer_sampler, max_iter, n_inner_step, n_v_step,
          lr_scheduler, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for _ in range(max_iter):
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
