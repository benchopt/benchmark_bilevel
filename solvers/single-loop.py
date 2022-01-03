
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit, prange


class Solver(BaseSolver):
    """Single loop Bi-level optimization algorithm."""
    name = 'single-loop'

    stopping_criterion = SufficientProgressCriterion(
        patience=100, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [1e-2],
        'batch_size, vr': [
            (1, 'saga'), (1, 'none'),  (32, 'none'), (64, 'none')
        ]
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 10

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
        v = np.zeros_like(inner_var)
        if self.step_size == 'auto':
            inner_step_size = 1 / self.f_inner.lipschitz_inner(
                inner_var, outer_var
            )
            outer_step_size = 1
        else:
            inner_step_size = self.step_size
            outer_step_size = self.step_size / 5

        eval_freq = 1000
        use_saga = self.vr == 'saga'
        if use_saga:
            memories = init_memory(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, v
            )
        else:
            memories = (np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1)))
        while callback((inner_var, outer_var)):
            inner_var, outer_var, v = saga(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, v,
                eval_freq, inner_step_size, outer_step_size,
                *memories, saga_inner=use_saga, saga_v=use_saga,
                batch_size=self.batch_size
            )
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


@njit(parallel=True)
def _init_memory(inner_oracle, outer_oracle, inner_var, outer_var, v):
    n_outer = outer_oracle.n_samples
    n_inner = inner_oracle.n_samples
    n_features = inner_oracle.n_features
    memory_inner_grad = np.zeros((n_inner + 1, n_features))
    memory_hvp = np.zeros((n_inner + 1, n_features))
    for id_inner in prange(n_inner):
        memory_inner_grad[id_inner] = inner_oracle.grad_inner_var(
            inner_var, outer_var, np.array([id_inner])
        )
        memory_hvp[id_inner] = inner_oracle.hvp(
            inner_var, outer_var, v, np.array([id_inner])
        )

    memory_outer_grad = np.zeros((n_outer + 1, n_features))
    for id_outer in prange(n_outer):
        memory_outer_grad[id_outer] = outer_oracle.grad_outer_var(
            inner_var, outer_var, np.array([id_outer])
        )

    return memory_inner_grad, memory_hvp, memory_outer_grad


def init_memory(inner_oracle, outer_oracle, inner_var, outer_var, v):
    memories = _init_memory(
        inner_oracle, outer_oracle, inner_var, outer_var, v
    )
    for mem in memories:
        mem[-1] = mem[:-1].mean(axis=0)
    return memories


@njit
def variance_reduction(grad, memory, idx):
    n_samples = memory.shape[0] - 1
    diff = grad - memory[idx[0]]
    direction = diff + memory[-1]
    memory[-1] += diff / n_samples
    memory[idx[0]] = grad
    return direction


@njit()
def saga(inner_oracle, outer_oracle, theta, lmbd, v,
         max_iter, inner_step_size, outer_step_size,
         memory_inner_grad, memory_hvp, memory_outer_grad,
         saga_inner=True, saga_v=True, batch_size=1):
    n_inner = inner_oracle.n_samples
    n_outer = outer_oracle.n_samples
    for i in range(max_iter):

        id_inner = np.array([
            np.random.randint(n_inner) for _ in range(batch_size)
        ])
        grad_inner_var = inner_oracle.grad_inner_var(theta, lmbd, id_inner)
        hvp = inner_oracle.hvp(theta, lmbd, v, id_inner)
        impl_grad = -inner_oracle.cross(theta, lmbd, v, id_inner)

        id_outer = np.array([
            np.random.randint(n_outer) for _ in range(batch_size)
        ])
        grad_outer = outer_oracle.grad_inner_var(theta, lmbd, id_outer)

        # here memory_*[-1] corresponds to the running average of
        # the gradients
        if saga_inner:
            grad_inner_var = variance_reduction(
                grad_inner_var, memory_inner_grad, id_inner
            )

        theta -= inner_step_size * grad_inner_var

        if saga_v:
            hvp = variance_reduction(
                hvp, memory_hvp, id_inner
            )
            grad_outer = variance_reduction(
                grad_outer, memory_outer_grad, id_outer
            )

        v -= inner_step_size * (hvp - grad_outer)

        lmbd -= outer_step_size * impl_grad
        theta, lmbd = inner_oracle.prox(theta, lmbd)
    return theta, lmbd, v
