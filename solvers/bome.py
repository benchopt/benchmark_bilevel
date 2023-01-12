from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from benchmark_utils import constants
    from benchmark_utils.sgd_inner import sgd_inner
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler


class Solver(BaseSolver):
    """Bilevel Optimization made easy."""
    name = 'BOME'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.01],
        'outer_ratio': [1.],
        'eval_freq': [1],
        'n_inner_step': [10],
        'phi_choice': ["grad", "value"],
        'eta': [.5]
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def skip(self, f_train, f_test, inner_var0, outer_var0, numba):
        if numba:
            return True, "numba is not useful for full bach solver."

        return False, None

    def set_objective(self, f_train, f_test, inner_var0, outer_var0, numba):

        self.f_inner = f_train
        self.f_outer = f_test

        self.MinibatchSampler = MinibatchSampler
        self.LearningRateScheduler = LearningRateScheduler

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.numba = numba

    def run(self, callback):
        eval_freq = self.eval_freq
        rng = np.random.RandomState(constants.RANDOM_STATE)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()

        # Init sampler and lr scheduler
        # It is a deterministic method
        batch_size_inner = self.f_inner.n_samples
        batch_size_outer = self.f_outer.n_samples

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

        while callback((inner_var, outer_var)):
            inner_var, outer_var = bome(
                self.f_inner, self.f_outer,
                inner_var, outer_var, inner_sampler, outer_sampler,
                eval_freq, lr_scheduler, self.n_inner_step, eta=self.eta,
                phi_choice=self.phi_choice,
                seed=rng.randint(constants.MAX_SEED)
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


def bome(inner_oracle, outer_oracle, inner_var, outer_var, inner_sampler,
         outer_sampler, max_iter, lr_scheduler, n_inner_step=10, eta=.5, 
         phi_choice="grad", seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_step_size, outer_step_size = lr_scheduler.get_lr()

        # Start algorithm
        inner_var_value = sgd_inner(
            inner_oracle, inner_var, outer_var, inner_step_size,
            inner_sampler=inner_sampler, n_inner_step=n_inner_step,
        )

        # Get inner gradient
        inner_slice, _ = outer_sampler.get_batch()
        grad_value_in, grad_value_out = inner_oracle.grad(
            inner_var, outer_var, inner_slice
        )
        grad_value_out -= inner_oracle.grad_outer_var(
            inner_var_value, outer_var, inner_slice
        )

        # Get outer gradient
        outer_slice, _ = outer_sampler.get_batch()
        grad_outer_in, grad_outer_out = outer_oracle.grad(
            inner_var, outer_var, outer_slice
        )

        norm_grad_value = np.linalg.norm(grad_value_in) ** 2 
        norm_grad_value += np.linalg.norm(grad_value_out) ** 2

        if phi_choice == "grad":
            phi = eta * norm_grad_value
        elif phi_choice == "value":
            phi = eta * (
                    inner_oracle.get_value(inner_var, outer_var)
                    - inner_oracle.get_value(inner_var_value, outer_var)
                )

        lmbda = max(
            (
                phi - np.dot(grad_outer_in, grad_value_in)
                - np.dot(grad_outer_out, grad_value_out)
            ) / norm_grad_value, 0
        )
        inner_var -= outer_step_size * (grad_outer_in + lmbda * grad_value_in)
        outer_var -= outer_step_size * (
            grad_outer_out + lmbda * grad_value_out
        )
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

    return inner_var, outer_var
