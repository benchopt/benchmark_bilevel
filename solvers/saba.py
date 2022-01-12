from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit, prange
    MinibatchSampler = import_ctx.import_from(
        'minibatch_sampler', 'MinibatchSampler'
    )
    LearningRateScheduler = import_ctx.import_from(
        'learning_rate_scheduler', 'LearningRateScheduler'
    )
    constants = import_ctx.import_from('constants')


class Solver(BaseSolver):
    """Stochastic Average Bi-level Algorithm."""
    name = 'SABA'

    stopping_criterion = SufficientProgressCriterion(
        patience=100, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': constants.STEP_SIZES,
        'outer_ratio': constants.OUTER_RATIOS,
        'batch_size, vr': [
            (1, 'saga')
        ]
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
        v = np.zeros_like(inner_var)

        # Init sampler and lr scheduler
        inner_sampler = MinibatchSampler(
            self.f_inner.numba_oracle, batch_size=self.batch_size
        )
        outer_sampler = MinibatchSampler(
            self.f_outer.numba_oracle, batch_size=self.batch_size
        )
        step_sizes = [self.step_size, self.step_size / self.outer_ratio]
        exponents = np.zeros(2)
        lr_scheduler = LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        # Init memory if needed
        use_saga = self.vr == 'saga'
        if use_saga:
            memories = init_memory(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, v
            )
            for mem in memories[:3]:
                inner_sampler.register_memory(mem)
            for mem in memories[3:]:
                outer_sampler.register_memory(mem)

        else:
            # To be compatible with numba compilation, memories need to always
            # be of type Array(ndim=2)
            memories = (
                np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1)),
                np.empty((1, 1))
            )

        # Start algorithm
        while callback((inner_var, outer_var)):
            inner_var, outer_var, v = saba(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, v, eval_freq,
                inner_sampler, outer_sampler, lr_scheduler,
                *memories, saga_inner=use_saga, saga_v=use_saga,
                seed=rng.randint(constants.MAX_SEED)
            )
            if np.isnan(outer_var).any():
                raise ValueError()
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


@njit
# @njit(parallel=True)
def _init_memory(inner_oracle, outer_oracle, inner_var, outer_var, v):
    n_outer = outer_oracle.n_samples
    n_inner = inner_oracle.n_samples
    n_features = inner_oracle.n_features
    memory_inner_grad = np.zeros((n_inner + 1, n_features))
    memory_hvp = np.zeros((n_inner + 1, n_features))
    memory_cross_v = np.zeros((n_inner + 1, n_features))
    for id_inner in prange(n_inner):
        _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, np.array([id_inner]), inverse='id'
        )
        memory_inner_grad[id_inner] = grad_inner_var
        memory_hvp[id_inner] = hvp
        memory_cross_v[id_inner] = cross_v

    memory_outer_grad = np.zeros((n_outer + 1, n_features))
    for id_outer in prange(n_outer):
        memory_outer_grad[id_outer] = outer_oracle.grad_outer_var(
            inner_var, outer_var, np.array([id_outer])
        )

    return memory_inner_grad, memory_hvp, memory_cross_v, memory_outer_grad


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
    diff = grad - memory[idx][0]
    direction = diff + memory[-1]
    memory[-1] += diff / n_samples
    memory[idx][0, :] = grad
    return direction


@njit
def saba(inner_oracle, outer_oracle, inner_var, outer_var, v, max_iter,
         inner_sampler, outer_sampler, lr_scheduler,
         memory_inner_grad, memory_hvp, memory_cross_v, memory_outer_grad,
         saga_inner=True, saga_v=True, saga_x=True, seed=None):
    np.random.seed(seed)
    for i in range(max_iter):
        inner_step_size, outer_step_size = lr_scheduler.get_lr()

        id_outer = outer_sampler.get_batch(outer_oracle)
        grad_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, id_outer
        )

        id_inner = inner_sampler.get_batch(inner_oracle)
        _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, id_inner, inverse='id'
        )

        # here memory_*[-1] corresponds to the running average of
        # the gradients
        if saga_inner:
            grad_inner_var = variance_reduction(
                grad_inner_var, memory_inner_grad, id_inner
            )

        inner_var -= inner_step_size * grad_inner_var

        if saga_v:
            hvp = variance_reduction(
                hvp, memory_hvp, id_inner
            )
            grad_outer = variance_reduction(
                grad_outer, memory_outer_grad, id_outer
            )

        v -= inner_step_size * (hvp - grad_outer)

        if saga_x:
            cross_v = variance_reduction(
                cross_v, memory_cross_v, id_inner
            )

        impl_grad -= cross_v
        outer_var -= outer_step_size * impl_grad

        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)
    return inner_var, outer_var, v
