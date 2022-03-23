from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit, prange
    constants = import_ctx.import_from('constants')
    MinibatchSampler = import_ctx.import_from(
        'minibatch_sampler', 'MinibatchSampler'
    )
    LearningRateScheduler = import_ctx.import_from(
        'learning_rate_scheduler', 'LearningRateScheduler'
    )


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

    def run(self, callback):
        eval_freq = constants.EVAL_FREQ  # // self.batch_size
        rng = np.random.RandomState(constants.RANDOM_STATE)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        v = np.zeros_like(inner_var)

        # Init sampler and lr scheduler
        inner_sampler = MinibatchSampler(
            self.f_inner.n_samples, batch_size=self.batch_size
        )
        outer_sampler = MinibatchSampler(
            self.f_outer.n_samples, batch_size=self.batch_size
        )
        step_sizes = np.array(
            [self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = np.zeros(2)
        lr_scheduler = LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        # Init memory if needed
        memories = init_memory(
            self.f_inner.numba_oracle, self.f_outer.numba_oracle,
            inner_var, outer_var, v, inner_sampler, outer_sampler
        )

        # Start algorithm
        while callback((inner_var, outer_var)):
            inner_var, outer_var, v = saba(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, v, eval_freq,
                inner_sampler, outer_sampler, lr_scheduler, *memories,
                seed=rng.randint(constants.MAX_SEED)
            )
            if np.isnan(outer_var).any():
                raise ValueError()
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


@njit
def init_memory(inner_oracle, outer_oracle, inner_var, outer_var, v,
                inner_sampler, outer_sampler):
    n_outer = outer_sampler.n_batches
    n_inner = inner_sampler.n_batches
    n_features = inner_oracle.n_features
    n_classes = inner_oracle.n_classes

    memory_inner_grad = np.zeros((n_inner + 1, n_features * n_classes))
    memory_hvp = np.zeros((n_inner + 1,  n_features * n_classes))
    memory_cross_v = np.zeros((n_inner + 1, n_features))
    for _ in prange(n_inner):
        slice_inner, (id_inner, weight) = inner_sampler.get_batch()
        _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, slice_inner
        )
        memory_inner_grad[id_inner, :] = grad_inner_var
        memory_inner_grad[-1, :] += weight * grad_inner_var
        memory_hvp[id_inner, :] = hvp
        memory_hvp[-1, :] += weight * hvp
        memory_cross_v[id_inner, :] = cross_v
        memory_cross_v[-1, :] += weight * cross_v

    memory_grad_in_outer = np.zeros((n_outer + 1, n_features * n_classes))
    for id_outer in prange(n_outer):
        slice_outer, (id_outer, weight) = outer_sampler.get_batch()
        memory_grad_in_outer[id_outer, :] = outer_oracle.grad_inner_var(
            inner_var, outer_var, slice_outer
        )
        memory_grad_in_outer[-1, :] += weight * memory_grad_in_outer[id_outer]

    return memory_inner_grad, memory_hvp, memory_cross_v, memory_grad_in_outer


@njit
def variance_reduction(grad, memory, vr_info):
    idx, weigth = vr_info
    diff = grad - memory[idx]
    direction = diff + memory[-1]
    memory[-1] += diff * weigth
    memory[idx, :] = grad
    return direction


@njit
def saba(inner_oracle, outer_oracle, inner_var, outer_var, v, max_iter,
         inner_sampler, outer_sampler, lr_scheduler,
         memory_inner_grad, memory_hvp, memory_cross_v, memory_grad_in_outer,
         seed=None):

    # Set seed for randomness
    np.random.seed(seed)

    for i in range(max_iter):
        inner_step_size, outer_step_size = lr_scheduler.get_lr()

        # Get all gradient for the batch
        slice_outer, vr_outer = outer_sampler.get_batch()
        grad_in_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )

        slice_inner, vr_inner = inner_sampler.get_batch()
        _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, slice_inner,
        )

        # Perform variance reduction on each oracle
        grad_inner_var = variance_reduction(
            grad_inner_var, memory_inner_grad, vr_inner
        )

        hvp = variance_reduction(hvp, memory_hvp, vr_inner)
        grad_in_outer = variance_reduction(
            grad_in_outer, memory_grad_in_outer, vr_outer
        )

        cross_v = variance_reduction(
            cross_v, memory_cross_v, vr_inner
        )

        inner_var -= inner_step_size * grad_inner_var
        v -= inner_step_size * (hvp - grad_in_outer)
        outer_var -= outer_step_size * cross_v

        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)
    return inner_var, outer_var, v
