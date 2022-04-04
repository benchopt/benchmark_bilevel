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
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [.01],
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

        # inner_var = self.f_inner.get_inner_var_star(outer_var)

        v = np.zeros_like(inner_var)
        # v = np.linalg.solve(
        #     self.f_inner.X.T.dot(self.f_inner.X)+np.exp(outer_var)*np.eye(self.f_inner.n_features),
        #     self.f_inner.X.T.dot(self.f_inner.y)
        # )

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

        # Init memory
        memories = init_memory(
            self.f_inner.numba_oracle, self.f_outer.numba_oracle,
            inner_var, outer_var, v, inner_sampler, outer_sampler
        )

        # Start algorithm
        i = 0
        while callback((inner_var, outer_var)):
            inner_var, outer_var, v, i = saba(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, v, eval_freq,
                inner_sampler, outer_sampler, lr_scheduler, *memories,
                seed=rng.randint(constants.MAX_SEED), i=i
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
    memory_inner_grad = np.zeros((n_inner + 1, n_features))
    memory_hvp = np.zeros((n_inner + 1, n_features))
    memory_cross_v = np.zeros((n_inner + 1, n_features))
    for _ in prange(n_inner):
        slice_inner, (id_inner, weight) = inner_sampler.get_batch()
        _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, slice_inner, inverse='id'
        )
        memory_inner_grad[id_inner, :] = grad_inner_var
        memory_inner_grad[-1, :] += weight * grad_inner_var
        memory_hvp[id_inner, :] = hvp
        memory_hvp[-1, :] += weight * hvp
        memory_cross_v[id_inner, :] = cross_v
        memory_cross_v[-1, :] += weight * cross_v

    memory_grad_in_outer = np.zeros((n_outer + 1, n_features))
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
         seed=None, i=0):

    # Set seed for randomness
    np.random.seed(seed)

    for _ in range(max_iter):
        i += 1
        inner_step_size, outer_step_size = lr_scheduler.get_lr()
        # if i >= 5075:
        #     inner_oracle.reg = 'lin'
        # if i == 5075:
        #     outer_var = np.exp(outer_var)

        # Get all gradient for the batch
        slice_outer, vr_outer = outer_sampler.get_batch()
        grad_in_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )

        slice_inner, vr_inner = inner_sampler.get_batch()
        _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, slice_inner, inverse='id'
        )

        # here memory_*[-1] corresponds to the running average of
        # the gradients
        grad_inner_var = variance_reduction(
            grad_inner_var, memory_inner_grad, vr_inner
        )


        inner_var -= inner_step_size * grad_inner_var
        # inner_var = inner_oracle.inner_var_star(outer_var, np.arange(8000))

        hvp = variance_reduction(hvp, memory_hvp, vr_inner)
        grad_in_outer = variance_reduction(
            grad_in_outer, memory_grad_in_outer, vr_outer
        )

        v -= inner_step_size * (hvp - grad_in_outer)
        # v = np.linalg.solve(inner_oracle.X.T.dot(inner_oracle.X) + np.exp(outer_var)*np.eye(200), grad_in_outer)

        cross_v = variance_reduction(
            cross_v, memory_cross_v, vr_inner
        )

        impl_grad -= cross_v
        outer_var -= outer_step_size * impl_grad

        if i >= 3250 * 2 ** 4:
            print('----------------')
            print("i", i)
            print("dir z", np.linalg.norm(inner_step_size * grad_inner_var))
            print("z", np.linalg.norm(inner_var))
            print("dir v", np.linalg.norm(inner_step_size * (hvp - grad_in_outer)))
            print("v", np.linalg.norm(v))
            print("dir x", np.linalg.norm(outer_step_size * impl_grad))
            print("x", np.linalg.norm(outer_var))
            print("exp(x)", np.linalg.norm(np.exp(outer_var)))

        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)
    return inner_var, outer_var, v, i
