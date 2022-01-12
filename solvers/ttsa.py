
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    MinibatchSampler = import_ctx.import_from(
        'minibatch_sampler', 'MinibatchSampler'
    )
    hia = import_ctx.import_from(
        'hessian_approximation', 'hia'
    )
    constants = import_ctx.import_from('constants')


class Solver(BaseSolver):
    """Single loop Bi-level optimization algorithm."""
    name = 'TTSA'

    stopping_criterion = SufficientProgressCriterion(
        patience=100, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': constants.STEP_SIZES,
        'outer_ratio': constants.OUTER_RATIOS,
        'n_hia_step': constants.N_HIA_STEPS,
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

    def run(self, callback):
        eval_freq = constants.EVAL_FREQ
        rng = np.random.RandomState(constants.RANDOM_STATE)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()

        # Init sampler and lr
        inner_sampler = MinibatchSampler(
            self.f_inner.numba_oracle, batch_size=self.batch_size
        )
        outer_sampler = MinibatchSampler(
            self.f_outer.numba_oracle, batch_size=self.batch_size
        )
        if self.step_size == 'auto':
            inner_step_size = 1 / self.f_inner.lipschitz_inner(
                inner_var, outer_var
            )
        else:
            inner_step_size = self.step_size
        hia_step = inner_step_size
        outer_step_size = inner_step_size / self.outer_ratio

        # Start algorithm
        eval_freq = constants.EVAL_FREQ
        while callback((inner_var, outer_var)):
            inner_var, outer_var, = ttsa(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var,
                eval_freq, inner_sampler, outer_sampler,
                inner_step_size, outer_step_size,
                self.n_hia_step, hia_step, seed=rng.randint(constants.MAX_SEED)
            )
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


@njit()
def ttsa(
    inner_oracle, outer_oracle, inner_var, outer_var, max_iter,
    inner_sampler, outer_sampler, inner_step_size, outer_step_size,
    n_hia_step, hia_step, seed=None
):
    """Numba compatible TTSA algorithm.

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
    inner_step_size: float
        Step size to update the inner variable.
    n_hia_step: int
        Maximal number of iteration for the HIA problem.
    hia_step_size: float
        Step size for the HIA sub-routine.
    inner_sampler, outer_sampler: MinibatchSampler
        Sampler to get minibatch in a fast and efficient way for the inner and
        outer problems.
    """

    np.random.seed(seed)

    for i in range(max_iter):

        # Step.1 - Update direction for z with momentum
        slice_inner, _ = inner_sampler.get_batch(inner_oracle)
        grad_inner_var = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice_inner
        )

        # Step.2 - Update the inner variable
        inner_var -= inner_step_size * grad_inner_var

        # Step.3 - Compute implicit grad approximation with HIA
        slice_outer, _ = outer_sampler.get_batch(outer_oracle)
        grad_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        ihvp = hia(
            inner_oracle, inner_var, outer_var, grad_outer,
            inner_sampler, n_hia_step, hia_step
        )
        impl_grad -= inner_oracle.cross(
            inner_var, outer_var, ihvp, slice_inner
        )

        # Step.4 - update the outer variables
        outer_var -= outer_step_size * impl_grad

        # Step.6 - project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)
    return inner_var, outer_var
