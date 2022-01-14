
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    MinibatchSampler = import_ctx.import_from(
        'minibatch_sampler', 'MinibatchSampler'
    )
    sgd_inner = import_ctx.import_from('sgd_inner', 'sgd_inner')
    sgd_v = import_ctx.import_from('hessian_approximation', 'sgd_v')
    constants = import_ctx.import_from('constants')


class Solver(BaseSolver):
    """Two loops solver."""
    name = 'amigo'

    stopping_criterion = SufficientProgressCriterion(
        patience=15, strategy='callback'
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
        return max(1, min(stop_val * 2, stop_val + 50))

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
        n_eval_freq = max(1, 1_024 // self.n_inner_step)
        inner_step_size = self.step_size
        outer_step_size = self.step_size / self.outer_ratio
        outer_var = self.outer_var0.copy()
        inner_var = self.inner_var0.copy()
        v = self.f_outer.grad_inner_var(inner_var, outer_var, np.array([0]))
        inner_sampler = MinibatchSampler(
            self.f_inner.n_samples, self.inner_batch_size
        )
        outer_sampler = MinibatchSampler(
            self.f_outer.n_samples, self.outer_batch_size
        )

        callback((inner_var, outer_var))
        # L = self.f_inner.lipschitz_inner(inner_var, outer_var)
        inner_var = sgd_inner(
            self.f_inner.numba_oracle, inner_var, outer_var,
            step_size=inner_step_size,
            inner_sampler=inner_sampler, n_inner_step=self.n_inner_step
        )
        while callback((inner_var, outer_var)):
            inner_var, outer_var, v = amigo(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, v, inner_sampler, outer_sampler,
                n_eval_freq, self.n_inner_step, 10,
                inner_step_size, outer_step_size, inner_step_size
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta

    def line_search(self, outer_var, grad):
        pass


@njit
def amigo(inner_oracle, outer_oracle, inner_var, outer_var, v,
          inner_sampler, outer_sampler,
          max_iter, n_inner_step, n_v_step,
          inner_step_size, outer_step_size, v_step_size):

    for i in range(max_iter):
        outer_slice, _ = outer_sampler.get_batch()
        grad_in, grad_out = outer_oracle.grad(
            inner_var, outer_var, outer_slice
        )

        v = sgd_v(inner_oracle, inner_var, outer_var, v, grad_in,
                  inner_sampler, n_v_step, v_step_size)

        inner_slice, _ = inner_sampler.get_batch()
        cross_hvp = inner_oracle.cross(inner_var, outer_var, v, inner_slice)
        implicit_grad = grad_out - cross_hvp

        outer_var -= outer_step_size * implicit_grad
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        inner_var = sgd_inner(
            inner_oracle, inner_var, outer_var, step_size=inner_step_size,
            inner_sampler=inner_sampler, n_inner_step=n_inner_step
        )
    return inner_var, outer_var, v
