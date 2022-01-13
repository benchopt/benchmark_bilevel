
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    MinibatchSampler = import_ctx.import_from(
        'minibatch_sampler', 'MinibatchSampler'
    )
    constants = import_ctx.import_from('constants')


class Solver(BaseSolver):
    """Single loop Bi-level optimization algorithm."""
    name = 'FSLA'

    stopping_criterion = SufficientProgressCriterion(
        patience=100, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': constants.STEP_SIZES,
        'outer_ratio': constants.OUTER_RATIOS,
        'batch_size': [1]
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
        eval_freq = constants.EVAL_FREQ // self.batch_size
        rng = np.random.RandomState(constants.RANDOM_STATE)

        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
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
        outer_step_size = inner_step_size / self.outer_ratio
        eta = inner_step_size

        # Init auxillary variables
        v = np.zeros_like(inner_var)
        d = np.zeros_like(outer_var)

        memory_inner = np.zeros_like(inner_var)
        memory_outer = np.zeros_like(outer_var)

        while callback((inner_var, outer_var)):
            inner_var, outer_var, v, d = fsla(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, v, d, memory_inner, memory_outer,
                eval_freq, inner_sampler, outer_sampler, inner_step_size,
                outer_step_size, eta, seed=rng.randint(constants.MAX_SEED)
            )
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


@njit()
def fsla(inner_oracle, outer_oracle, inner_var, outer_var, v, d, memory_inner,
         memory_outer, max_iter, inner_sampler, outer_sampler,
         inner_step_size, outer_step_size, eta, seed=None):
    # Set seed for randomness
    np.random.seed(seed)
    for i in range(max_iter):

        # Step.1 - SGD step on the inner problem
        slice_inner, _ = inner_sampler.get_batch(inner_oracle)
        grad_inner_var = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice_inner
        )
        inner_var -= inner_step_size * grad_inner_var

        # Step.2 - SGD step on the auxillary variable v
        slice_inner2, _ = inner_sampler.get_batch(inner_oracle)
        hvp = inner_oracle.hvp(inner_var, outer_var, v, slice_inner2)
        slice_outer, _ = outer_sampler.get_batch(outer_oracle)
        grad_outer = outer_oracle.grad_inner_var(
            inner_var, outer_var, slice_outer
        )
        v_old = v.copy()
        v -= inner_step_size * (hvp - grad_outer)

        # Step.3 - compute the implicit gradient estimate
        slice_outer2, _ = outer_sampler.get_batch(outer_oracle)
        impl_grad = outer_oracle.grad_outer_var(
            inner_var, outer_var, slice_outer2
        )
        impl_grad_old = outer_oracle.grad_outer_var(
            memory_inner, memory_outer, slice_outer2
        )
        slice_inner3, _ = inner_sampler.get_batch(inner_oracle)
        impl_grad -= inner_oracle.cross(inner_var, outer_var, v, slice_inner3)
        impl_grad_old -= inner_oracle.cross(
            memory_inner, memory_outer, v_old, slice_inner3
        )

        # Step.4 - update direction with momentum
        d = impl_grad + (1-eta) * (d - impl_grad_old)

        # Step.5 - update the outer variable
        outer_var -= outer_step_size * d

        # Step.6 - project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        # Step.7 - update memories
        memory_inner = inner_var
        memory_outer = outer_var

    return inner_var, outer_var, v, d
