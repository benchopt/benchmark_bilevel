from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import jax
    import jax.numpy as jnp
    from functools import partial

    from benchmark_utils import constants
    from benchmark_utils.gd_inner import gd_inner_jax
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler


class Solver(BaseSolver):
    """Bilevel Optimization Made Easy (BOME).

    M. Ye, B. Liu, S. Wright, P. Stone and Q. Liu, "BOME! Bilevel Optimization
    Made Easy: A Simple First-Order Approach", NeurIPS 2022."""
    name = 'BOME'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'random_state': [1],
        'choice_phi': ["grad_norm"],
        'eta': [5e-1],
        'n_inner_steps': [10],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_inner, f_outer, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0, f_inner_fb, f_outer_fb,):

        self.f_inner = f_inner_fb
        self.f_outer = f_outer_fb

        self.inner_var = inner_var0
        self.outer_var = outer_var0
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        self.run_once(2)

    def run(self, callback):
        self.inner_var = self.inner_var0
        self.outer_var = self.outer_var0

        # Init lr scheduler
        step_sizes = jnp.array(
            [self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = jnp.array([0, 0])
        state_lr = init_lr_scheduler(step_sizes, exponents)

        grad_inner = jax.jit(jax.grad(self.f_inner, argnums=(0, 1)))
        grad_outer = jax.jit(jax.grad(self.f_outer, argnums=(0, 1)))
        inner_bome = partial(
                gd_inner_jax,
                grad_inner=lambda y, x: grad_inner(y, x)[0],
                n_steps=self.n_inner_steps
        )

        # Start algorithm
        while callback():
            step_sizes, state_lr = update_lr(
                state_lr
            )
            lr_inner, lr_outer = step_sizes

            # Run the inner procedure
            inner_var_star = inner_bome(self.inner_var,
                                        self.outer_var,
                                        lr_inner)

            # Compute oracles
            grad_outer_inner_var, grad_outer_outer_var = grad_outer(
                self.inner_var, self.outer_var
            )
            grad_q_inner_var, grad_q_outer_var = grad_inner(
                self.inner_var, self.outer_var
            )
            grad_q_outer_var -= grad_inner(
                inner_var_star, self.outer_var
            )[1]

            # Compute phi and lmbda
            squared_norm_grad_q = jnp.linalg.norm(grad_q_inner_var)**2
            squared_norm_grad_q += jnp.linalg.norm(grad_q_outer_var)**2
            if self.choice_phi == 'grad_norm':
                phi = squared_norm_grad_q
            else:
                phi = (
                        self.f_inner(self.inner_var, self.outer_var)
                        - self.f_inner(inner_var_star, self.outer_var)
                )
            phi *= self.eta
            dot_grad = jnp.dot(grad_outer_inner_var, grad_q_inner_var)
            dot_grad += jnp.dot(grad_outer_outer_var, grad_q_outer_var)
            lmbda = jnp.maximum(phi - dot_grad, 0) / squared_norm_grad_q

            # Compute the update direction of the inner and outer variables
            d_inner = grad_outer_inner_var + lmbda * grad_q_inner_var
            d_outer = grad_outer_outer_var + lmbda * grad_q_outer_var

            # Update inner and outer variables
            self.inner_var -= lr_inner * d_inner
            self.outer_var -= lr_outer * d_outer

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var)
