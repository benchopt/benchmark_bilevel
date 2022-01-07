
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from oracles.minibatch_sampler import MinibatchSampler


class Solver(BaseSolver):
    """Single loop Bi-level optimization algorithm."""
    name = 'MRBO'

    stopping_criterion = SufficientProgressCriterion(
        patience=100, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [1e-1, 1e-2],
        'outer_ratio': [2, 5],
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
        eta = inner_step_size
        hia_step = inner_step_size
        n_hia_step = 10

        memory_inner = np.zeros((2, *inner_var.shape), inner_var.dtype)
        memory_outer = np.zeros((2, *outer_var.shape), outer_var.dtype)

        eval_freq = 1024
        while callback((inner_var, outer_var)):
            inner_var, outer_var, memory_inner, memory_outer = sustain(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, memory_inner, memory_outer,
                eval_freq, inner_sampler, outer_sampler,
                inner_step_size, outer_step_size,
                n_hia_step, hia_step, eta
            )
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


@njit
def joint_shia(
    inner_oracle, inner_var, outer_var, v, inner_var_old, outer_var_old, v_old,
    inner_sampler, n_step, step_size
):
    """Hessian Inverse Approximation subroutine from [Ji2021].

    This implement Algorithm.3
    """
    s = v
    s_old = v_old
    for i in range(n_step):
        inner_slice, _ = inner_sampler.get_batch(inner_oracle)
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * hvp
        s += v
        hvp_old = inner_oracle.hvp(
            inner_var_old, outer_var_old, v_old, inner_slice
        )
        v_old -= step_size * hvp_old
        s_old += v_old
    return step_size * v, step_size * v_old


@njit()
def sustain(inner_oracle, outer_oracle, inner_var, outer_var,
            memory_inner, memory_outer, max_iter, inner_sampler, outer_sampler,
            inner_step_size, outer_step_size, n_hia_step, hia_step, eta):
    for i in range(max_iter):

        # Step.1 - Update direction for z with momentum
        slice_inner, _ = inner_sampler.get_batch(inner_oracle)
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
        slice_outer, _ = outer_sampler.get_batch(outer_oracle)
        grad_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        grad_outer_old, impl_grad_old = outer_oracle.grad(
            memory_inner[0], memory_outer[0], slice_outer
        )
        ihvp, ihvp_old = joint_shia(
            inner_oracle, inner_var, outer_var, grad_outer,
            memory_inner[0], memory_outer[0], grad_outer_old,
            inner_sampler, n_hia_step, hia_step
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
        inner_var -= inner_step_size * memory_inner[1]
        outer_var -= outer_step_size * memory_outer[1]

        # Step.6 - project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)
    return inner_var, outer_var, memory_inner, memory_outer
