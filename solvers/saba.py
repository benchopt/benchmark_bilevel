from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit, prange
    from numba.experimental import jitclass
    constants = import_ctx.import_from('constants')
    MinibatchSampler = import_ctx.import_from(
        'minibatch_sampler', 'MinibatchSampler'
    )
    spec_minibatch_sampler = import_ctx.import_from(
        'minibatch_sampler', 'spec'
    )
    LearningRateScheduler = import_ctx.import_from(
        'learning_rate_scheduler', 'LearningRateScheduler'
    )
    spec_scheduler = import_ctx.import_from(
        'learning_rate_scheduler', 'spec'
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
        'outer_ratio': [1.],
        'batch_size': [64],
        'eval_freq': [1],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_train, f_test, inner_var0, outer_var0, numba):
        if self.batch_size == 'full':
            numba = False
        if numba:
            self.f_inner = f_train.numba_oracle
            self.f_outer = f_test.numba_oracle

            def init_memory(init_memory_fb):
                def f(*args, **kwargs):
                    return njit(_init_memory)(init_memory_fb, *args, **kwargs)
                return f
            self.init_memory = init_memory(njit(_init_memory_fb))
            self.MinibatchSampler = jitclass(MinibatchSampler,
                                             spec_minibatch_sampler)

            self.LearningRateScheduler = jitclass(LearningRateScheduler,
                                                  spec_scheduler)

            def saba(variance_reduction):
                def f(*args, **kwargs):
                    return njit(_saba)(variance_reduction, *args, **kwargs)
                return f
            self.saba = saba(njit(variance_reduction))
        else:
            self.f_inner = f_train
            self.f_outer = f_test

            def init_memory(init_memory_fb):
                def f(*args, **kwargs):
                    return _init_memory(init_memory_fb, *args, **kwargs)
                return f

            def saba(variance_reduction):
                def f(*args, **kwargs):
                    return _saba(variance_reduction, *args, **kwargs)
                return f
            self.saba = saba(variance_reduction)
            self.init_memory = init_memory(_init_memory_fb)
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.numba = numba

    def run(self, callback):
        eval_freq = self.eval_freq  # // self.batch_size
        rng = np.random.RandomState(constants.RANDOM_STATE)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        v = np.zeros_like(inner_var)

        # Init sampler and lr scheduler
        if self.batch_size == 'full':
            batch_size_inner = self.f_inner.n_samples
            batch_size_outer = self.f_outer.n_samples
        else:
            batch_size_inner = self.batch_size
            batch_size_outer = self.batch_size
        inner_sampler = self.MinibatchSampler(
            self.f_inner.n_samples, batch_size=batch_size_inner
        )
        outer_sampler = self.MinibatchSampler(
            self.f_outer.n_samples, batch_size=batch_size_outer
        )
        step_sizes = np.array(
            [self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = np.zeros(2)
        lr_scheduler = self.LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        # Init memory if needed
        memory_inner_grad, memory_hvp, memory_cross_v, \
            memory_grad_in_outer = self.init_memory(
                self.f_inner, self.f_outer,
                inner_var, outer_var, v, inner_sampler, outer_sampler
            )

        # Start algorithm
        while callback((inner_var, outer_var)):
            inner_var, outer_var, v = self.saba(
                self.f_inner, self.f_outer,
                inner_var, outer_var, v, eval_freq,
                inner_sampler, outer_sampler, lr_scheduler, memory_inner_grad,
                memory_hvp, memory_cross_v, memory_grad_in_outer,
                seed=rng.randint(constants.MAX_SEED)
            )
            if np.isnan(outer_var).any():
                raise ValueError()
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


def _init_memory(
    _init_memory_fb,
    inner_oracle,
    outer_oracle,
    inner_var,
    outer_var,
    v,
    inner_sampler,
    outer_sampler,
    mode="zero",
):
    if mode == "full":
        memories = _init_memory_fb(
            inner_oracle,
            outer_oracle,
            inner_var,
            outer_var,
            v,
            inner_sampler,
            outer_sampler,
        )
        for mem in memories:
            mem[-1] = mem[:-1].sum(axis=0) / mem[:-1].shape[0]
    else:
        n_outer = outer_sampler.n_batches
        n_inner = inner_sampler.n_batches
        inner_size, outer_size = inner_oracle.variables_shape
        memories = (
            np.zeros((n_inner + 1, inner_size[0])),
            np.zeros((n_inner + 1, inner_size[0])),
            np.zeros((n_inner + 1, outer_size[0])),
            np.zeros((n_outer + 1, inner_size[0])),
        )
    return memories


def _init_memory_fb(inner_oracle, outer_oracle, inner_var, outer_var, v,
                    inner_sampler, outer_sampler):
    n_outer = outer_sampler.n_batches
    n_inner = inner_sampler.n_batches
    inner_var_shape, outer_var_shape = inner_oracle.variables_shape.ravel()
    memory_inner_grad = np.zeros((n_inner + 1, inner_var_shape))
    memory_hvp = np.zeros((n_inner + 1, inner_var_shape))
    memory_cross_v = np.zeros((n_inner + 1, outer_var_shape))
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

    memory_grad_in_outer = np.zeros((n_outer + 1, inner_var_shape))
    for id_outer in prange(n_outer):
        slice_outer, (id_outer, weight) = outer_sampler.get_batch()
        memory_grad_in_outer[id_outer, :] = outer_oracle.grad_inner_var(
            inner_var, outer_var, slice_outer
        )
        memory_grad_in_outer[-1, :] += weight * memory_grad_in_outer[id_outer]

    return memory_inner_grad, memory_hvp, memory_cross_v, memory_grad_in_outer


def variance_reduction(grad, memory, vr_info):
    idx, weigth = vr_info
    diff = grad - memory[idx]
    direction = diff + memory[-1]
    memory[-1] += diff * weigth
    memory[idx, :] = grad
    return direction


def _saba(variance_reduction, inner_oracle, outer_oracle, inner_var, outer_var,
          v, max_iter, inner_sampler, outer_sampler, lr_scheduler,
          memory_inner_grad, memory_hvp, memory_cross_v, memory_grad_in_outer,
          seed=None):

    # Set seed for randomness
    if seed is not None:
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
            inner_var, outer_var, v, slice_inner, inverse='id'
        )

        # here memory_*[-1] corresponds to the running average of
        # the gradients

        grad_inner_var = variance_reduction(
            grad_inner_var, memory_inner_grad, vr_inner
        )
        # import ipdb; ipdb.set_trace()
        inner_var -= inner_step_size * grad_inner_var

        hvp = variance_reduction(hvp, memory_hvp, vr_inner)
        grad_in_outer = variance_reduction(
            grad_in_outer, memory_grad_in_outer, vr_outer
        )

        v -= inner_step_size * (hvp - grad_in_outer)

        cross_v = variance_reduction(
            cross_v, memory_cross_v, vr_inner
        )
        impl_grad -= cross_v

        outer_var -= outer_step_size * impl_grad

        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)
    return inner_var, outer_var, v
