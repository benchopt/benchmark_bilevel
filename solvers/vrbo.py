from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    constants = import_ctx.import_from('constants')
    MinibatchSampler = import_ctx.import_from(
        'minibatch_sampler', 'MinibatchSampler'
    )
    LearningRateScheduler = import_ctx.import_from(
        'learning_rate_scheduler', 'LearningRateScheduler'
    )
    joint_shia = import_ctx.import_from(
        'hessian_approximation', 'joint_shia'
    )
    shia = import_ctx.import_from(
        'hessian_approximation', 'shia'
    )
    sgd_inner_vrbo = import_ctx.import_from(
        'sgd_inner', 'sgd_inner_vrbo'
    )


class Solver(BaseSolver):
    """Single loop Bi-level optimization algorithm."""
    name = 'VRBO'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': constants.STEP_SIZES,
        'outer_ratio': constants.OUTER_RATIOS,
        'n_hia_step': constants.N_HIA_STEPS,
        'batch_size': constants.BATCH_SIZES,
        'period': [3],
        'n_inner_step': constants.N_INNER_STEPS,
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
        memory_inner = np.zeros((2, *inner_var.shape), inner_var.dtype)
        memory_outer = np.zeros((2, *outer_var.shape), outer_var.dtype)

        # Init sampler and lr scheduler
        inner_sampler = MinibatchSampler(
            self.f_inner.n_samples, batch_size=self.batch_size
        )
        outer_sampler = MinibatchSampler(
            self.f_outer.n_samples, batch_size=self.batch_size
        )
        step_sizes = np.array(  # (inner_ss, hia_lr, outer_ss)
            [
                self.step_size,
                self.step_size,
                self.step_size / self.outer_ratio,
            ]
        )
        exponents = np.zeros(3)
        lr_scheduler = LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        eval_freq = constants.EVAL_FREQ
        # Start algorithm
        while callback((inner_var, outer_var)):
            inner_var, outer_var, memory_inner, memory_outer = vrbo(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, memory_inner, memory_outer,
                eval_freq, inner_sampler, outer_sampler,
                lr_scheduler, self.n_hia_step, self.n_inner_step, self.period,
                seed=rng.randint(constants.MAX_SEED)
            )
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


@njit()
def vrbo(inner_oracle, outer_oracle, inner_var, outer_var,
         memory_inner, memory_outer, max_iter, inner_sampler, outer_sampler,
         lr_scheduler, n_hia_step, n_inner_steps, period, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_lr, hia_lr, outer_lr = lr_scheduler.get_lr()

        # Step.1 - (Re)initialize directions for z and x
        if i % period == 0:
            slice_inner, _ = inner_sampler.get_batch()
            grad_inner_var = inner_oracle.grad_inner_var(
                inner_var, outer_var, slice_inner
            )
            memory_inner[1] = grad_inner_var

            slice_outer, _ = outer_sampler.get_batch()
            grad_outer, impl_grad = outer_oracle.grad(
                inner_var, outer_var, slice_outer
            )
            ihvp = shia(
                inner_oracle, inner_var, outer_var, grad_outer, inner_sampler,
                n_hia_step, hia_lr
            )
            impl_grad -= inner_oracle.cross(
                inner_var, outer_var, ihvp, slice_inner
            )
            memory_outer[1] = impl_grad

        # Step.2 - Update outer variable and memory
        memory_outer[0] = outer_var
        outer_var -= outer_lr * memory_outer[1]

        # Step.3 - Project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        inner_var, outer_var, memory_inner, memory_outer = sgd_inner_vrbo(
            inner_oracle, outer_oracle, inner_var, outer_var, inner_lr,
            inner_sampler, outer_sampler, n_inner_steps, memory_inner,
            memory_outer, n_hia_step, hia_lr
        )

    return inner_var, outer_var, memory_inner, memory_outer
