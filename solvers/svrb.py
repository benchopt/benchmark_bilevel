
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from oracles.minibatch_sampler import MinibatchSampler


class Solver(BaseSolver):
    """Single loop Bi-level optimization algorithm."""
    name = 'SVRB'

    stopping_criterion = SufficientProgressCriterion(
        patience=100, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [1e-2],
        'outer_ratio': [50],
        'batch_size': [1, 32]
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 50

    def set_objective(self, f_train, f_test, inner_var0, outer_var0):
        self.f_inner = f_train
        self.f_outer = f_test
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.random_state = 29

    def run(self, callback):
        # rng = np.random.RandomState(self.random_state)
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
        eta = inner_step_size * np.ones(5)

        memory_inner = np.zeros((2, *inner_var.shape), inner_var.dtype)
        memory_hessian = np.zeros(
            (*inner_var.shape, *inner_var.shape),
            inner_var.dtype
        )
        memory_cross = np.zeros(
            (*outer_var.shape, *inner_var.shape), inner_var.dtype
        )

        memory_outer = np.zeros((2, *outer_var.shape), outer_var.dtype)
        memory_outer2 = np.zeros(*inner_var.shape, inner_var.dtype)

        eval_freq = 1024
        while callback((inner_var, outer_var)):
            inner_var, outer_var, memory_inner, memory_outer = svrb(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, memory_inner, memory_hessian,
                memory_cross, memory_outer, memory_outer2,
                eval_freq, inner_sampler, outer_sampler,
                inner_step_size, outer_step_size, eta
            )
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


@njit()
def svrb(inner_oracle, outer_oracle, inner_var, outer_var,
         memory_inner, memory_hessian, memory_cross,
         memory_outer, memory_outer2,
         max_iter, inner_sampler, outer_sampler,
         inner_step_size, outer_step_size, eta):
    for i in range(max_iter):

        # Step.1 - Draw batches
        slice_inner, _ = inner_sampler.get_batch(inner_oracle)
        slice_outer, _ = outer_sampler.get_batch(outer_oracle)

        # Step.2 - Update direction for z with momentum
        grad_inner_var = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice_inner
        )
        grad_inner_var_old = inner_oracle.grad_inner_var(
            memory_inner[0], memory_outer[0], slice_inner
        )
        memory_inner[1] = eta[0] * grad_inner_var + (1-eta[0]) * (
            memory_inner[1] + grad_inner_var - grad_inner_var_old
        )

        # Step.3 - Update estimate of outer_grad
        outer_grad_inner, outer_grad_outer = outer_oracle.grad(
                inner_var, outer_var, slice_outer
        )

        outer_grad_inner_old, outer_grad_outer_old = outer_oracle.grad(
            memory_inner[0], memory_outer[0], slice_outer
        )
        memory_outer2 = (1-eta[1])*(memory_outer2 - outer_grad_inner_old)
        memory_outer2 += outer_grad_inner

        memory_outer[1] = (1-eta[2])*(memory_outer[1] - outer_grad_outer_old)
        memory_outer[1] += outer_grad_outer

        hessian = inner_oracle.hessian(inner_var, outer_var, slice_inner)
        hessian_old = inner_oracle.hessian(
            memory_inner[0], memory_outer[0], slice_inner
        )
        memory_hessian = (1-eta[3])*(memory_hessian - hessian_old) + hessian

        cross_matrix = inner_oracle.cross_matrix(
            inner_var, outer_var, slice_inner
        )
        cross_matrix_old = inner_oracle.cross_matrix(
            memory_inner[0], memory_outer[0], slice_inner
        )
        memory_cross = (1-eta[4])*(memory_cross - cross_matrix_old)
        memory_cross += cross_matrix

        impl_grad = - memory_cross.dot(np.linalg.solve(
            memory_hessian, memory_outer2
        ))

        impl_grad += memory_outer[2]

        # Step.4 - Save the current variables
        memory_inner[0] = inner_var
        memory_outer[0] = outer_var

        # Step.5 - update the variables with the directions
        inner_var -= inner_step_size * memory_inner[1]
        outer_var -= outer_step_size * impl_grad

        # Step.6 - project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)
    return inner_var, outer_var, memory_inner, memory_outer
