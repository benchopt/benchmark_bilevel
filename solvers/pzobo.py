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
    """Partial Zeroth-Order-like Bilevel Optimizer (PZOBO).

    D. Sow, K. Ji and Y. Liang. "On the Convergence Theory for Hessian-Free
    Bilevel Algorithms". arxiv:2110.07004 2022"""
    name = 'PZOBO'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'random_state': [1],
        'mu': [.1],
        'n_inner_steps': [10],
        'n_gaussian_vectors': [10],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_inner, f_outer, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):
        self.f_inner = partial(f_inner, start=0, batch_size=n_inner_samples)
        self.f_outer = partial(f_outer, start=0, batch_size=n_outer_samples)

        self.inner_loop = partial(
            gd_inner_jax,
            grad_inner=jax.grad(self.f_inner),
        )

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        self.run_once(2)

    def run(self, callback):
        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()

        # Init lr scheduler
        step_sizes = jnp.array(
            [self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = jnp.array(
            [0., 0.]
        )
        state_lr = init_lr_scheduler(step_sizes, exponents)
        key = jax.random.PRNGKey(self.random_state)
        grad_outer = jax.jit(jax.grad(self.f_outer, argnums=(0, 1)))

        inner_pzobo = partial(
                gd_inner_jax,
                grad_inner=jax.jit(jax.grad(self.f_inner, argnums=0)),
                n_steps=self.n_inner_steps
        )
        vmapped_inner = jax.jit(jax.vmap(
            inner_pzobo,
            in_axes=(None, 0, None)
        ))

        # Start algorithm
        while callback():
            key = jax.random.split(key, 1)[0]
            (inner_step_size, outer_step_size), state_lr = update_lr(
                state_lr
            )

            inner_var_old = self.inner_var.copy()

            # Update inner variable by GD
            self.inner_var = inner_pzobo(self.inner_var, self.outer_var,
                                         inner_step_size)

            # Generate Gaussian vectors
            U = jax.random.normal(key, (self.n_gaussian_vectors,
                                        self.outer_var.shape[0]))

            # Perturbate the outer variable in random directions
            outer_var_aux = self.outer_var + self.mu * U
            deltas = vmapped_inner(inner_var_old, outer_var_aux,
                                   inner_step_size)
            deltas -= self.inner_var
            deltas /= self.mu
            grad_outer_in, grad_outer_out = grad_outer(self.inner_var,
                                                       self.outer_var)
            es_estimator = U.T.dot(deltas @ grad_outer_in)
            es_estimator /= self.n_gaussian_vectors
            self.outer_var -= outer_step_size * (es_estimator + grad_outer_out)

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var)
