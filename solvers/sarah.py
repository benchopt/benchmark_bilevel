from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    constants = import_ctx.import_from('constants')
    MinibatchSampler = import_ctx.import_from(
        'minibatch_sampler', 'MinibatchSampler'
    )
    LearningRateScheduler = import_ctx.import_from(
        'learning_rate_scheduler', 'LearningRateScheduler'
    )


class Solver(BaseSolver):
    """Adaptation of SARAH for bilevel optimization"""
    name = 'sarah'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': np.logspace(-5, 3, 9, base=2),
        'outer_ratio': np.logspace(-2, 1, 6),
        'batch_size': [64],
        'period': np.logspace(1, 6, 6),
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
        step_sizes = np.array(  # (inner_ss, hia_lr, outer_ss)
            [
                self.step_size,
                self.step_size / self.outer_ratio,
            ]
        )
        exponents = np.zeros(2)
        lr_scheduler = LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        eval_freq = constants.EVAL_FREQ

        inner_var_old = None
        outer_var_old = None
        v_old = None
        d_inner = None
        d_v = None
        d_outer = None
        i_min = 0
        # Start algorithm
        while callback((inner_var, outer_var)):
            inner_var, outer_var, inner_var_old, v_old, outer_var_old,\
                d_inner, d_v, d_outer, i_min = sarah(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v,
                    eval_freq, inner_sampler, outer_sampler,
                    lr_scheduler, inner_var_old=inner_var_old, v_old=v_old,
                    outer_var_old=outer_var_old, d_inner=d_inner, d_v=d_v,
                    d_outer=d_outer, i_min=i_min, period=self.period,
                    seed=rng.randint(constants.MAX_SEED)
                )
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


def sarah(inner_oracle, outer_oracle, inner_var, outer_var, v, max_iter,
          inner_sampler, outer_sampler, lr_scheduler, inner_var_old=None,
          v_old=None, outer_var_old=None, d_inner=None, d_v=None, d_outer=None,
          i_min=0, period=100, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(i_min, i_min+max_iter):
        inner_lr, outer_lr = lr_scheduler.get_lr()
        if i % period == 0:
            _, d_inner, hvp, cross_v = inner_oracle.get_oracles(inner_var,
                                                                outer_var, v)
            grad_in, d_outer = outer_oracle.get_grad(inner_var, outer_var)

            d_v = hvp - grad_in
            d_outer -= cross_v

            inner_var_old = inner_var.copy()
            v_old = v.copy()
            outer_var_old = outer_var.copy()

            inner_var -= inner_lr * d_inner
            v -= inner_lr * d_v
            outer_var -= outer_lr * d_outer

        else:
            slice_inner, _ = inner_sampler.get_batch()
            _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
                inner_var, outer_var, v, slice_inner, inverse='id'
            )
            _, grad_inner_var_old, hvp_old, cross_v_old = inner_oracle.oracles(
                inner_var_old, outer_var_old, v_old, slice_inner, inverse='id'
            )

            slice_outer, _ = outer_sampler.get_batch()
            grad_in_outer, grad_out_outer = outer_oracle.grad(
                inner_var, outer_var, slice_outer
            )
            grad_in_outer_old, grad_out_outer_old = outer_oracle.grad(
                inner_var_old, outer_var_old, slice_outer
            )

            d_inner += grad_inner_var - grad_inner_var_old
            d_v += (hvp - hvp_old) - (grad_in_outer - grad_in_outer_old)
            d_outer += (grad_out_outer - grad_out_outer_old)
            d_outer -= (cross_v - cross_v_old)

            inner_var_old = inner_var.copy()
            v_old = v.copy()
            outer_var_old = outer_var.copy()

            inner_var -= inner_lr * d_inner
            v -= inner_lr * d_v
            outer_var -= outer_lr * d_outer

    return (
        inner_var, outer_var, inner_var_old, v_old, outer_var_old, d_inner,
        d_v, d_outer, i_min+max_iter
    )
