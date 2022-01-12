
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    MinibatchSampler = import_ctx.import_from(
        'minibatch_sampler', 'MinibatchSampler'
    )
    shia = import_ctx.import_from('hessian_approximation', 'shia')
    sgd_inner = import_ctx.import_from('sgd_inner', 'sgd_inner')
    constants = import_ctx.import_from('constants')


class Solver(BaseSolver):
    """StocBio solver from [Ji2021]

    :cat:`two-loops`
    """
    name = 'stocBiO'

    stopping_criterion = SufficientProgressCriterion(
        patience=100, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': constants.STEP_SIZES,
        'outer_ratio': constants.OUTER_RATIOS,
        'n_inner_step': constants.N_INNER_STEPS,
        'batch_size': [1],
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
        outer_var = self.outer_var0.copy()
        inner_var = self.inner_var0.copy()

        # Init sampler and lr
        shia_step_size = 0.5
        inner_step_size = self.step_size
        outer_step_size = self.step_size / self.outer_ratio
        inner_sampler = MinibatchSampler(
            self.f_inner.numba_oracle, self.inner_batch_size
        )
        outer_sampler = MinibatchSampler(
            self.f_outer.numba_oracle, self.outer_batch_size
        )

        # Start algorithm
        callback((inner_var, outer_var))
        # L = self.f_inner.lipschitz_inner(inner_var, outer_var)
        inner_var = sgd_inner(
            self.f_inner.numba_oracle, inner_var, outer_var,
            step_size=inner_step_size,
            inner_sampler=inner_sampler, n_inner_step=self.n_inner_step
        )
        while callback((inner_var, outer_var)):
            inner_var, outer_var = stocbio(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, eval_freq, outer_step_size,
                self.n_inner_step, inner_step_size,
                n_shia_step=self.n_inner_step, shia_step_size=shia_step_size,
                inner_sampler=inner_sampler, outer_sampler=outer_sampler,
                seed=rng.randint(constants.MAX_SEED)
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta

    def line_search(self, outer_var, grad):
        pass


@njit
def stocbio(inner_oracle, outer_oracle, inner_var, outer_var,
            max_iter, outer_step_size, n_inner_step, inner_step_size,
            n_shia_step, shia_step_size, inner_sampler, outer_sampler,
            seed=None):
    """Numba compatible stocBiO algorithm.

    Parameters
    ----------
    inner_oracle, outer_oracle: NumbaOracle
        Inner and outer problem oracles used to compute gradients, etc...
    inner_var, outer_var: ndarray
        Current estimates of the inner and outer variables of the bi-level
        problem.
    max_iter: int
        Maximal number of iterations for the outer problem.
    outer_step_size: float
        Step size to update the outer variable.
    n_inner_step: int
        Maximal number of iterations for the inner problem.
    inner_step_size: float
        Step size to update the inner variable.
    n_shia_step: int
        Maximal number of iterations for the shia problem.
    shia_step_size: float
        Step size for the shia sub-routine.
    inner_sampler, outer_sampler: MinibatchSampler
        Sampler to get minibatch in a fast and efficient way for the inner and
        outer problems.
    """

    np.random.seed(seed)

    for i in range(max_iter):
        outer_slice, _ = outer_sampler.get_batch(outer_oracle)
        grad_in, grad_out = outer_oracle.grad(
            inner_var, outer_var, outer_slice
        )

        implicit_grad = shia(
            inner_oracle, inner_var, outer_var, grad_in,
            inner_sampler, n_shia_step, shia_step_size
        )
        inner_slice, _ = inner_sampler.get_batch(inner_oracle)
        implicit_grad = inner_oracle.cross(
            inner_var, outer_var, implicit_grad, inner_slice
        )
        grad_outer_var = grad_out - implicit_grad

        outer_var -= outer_step_size * grad_outer_var
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        inner_var = sgd_inner(
            inner_oracle, inner_var, outer_var, step_size=inner_step_size,
            inner_sampler=inner_sampler, n_inner_step=n_inner_step
        )
    return inner_var, outer_var
