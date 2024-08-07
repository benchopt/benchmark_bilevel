from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils import constants
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp
    from functools import partial

    import jaxopt


class Solver(BaseSolver):
    """Gradient descent with JAXopt solvers.

    M. Blondel, Q. Berthet, M. Cuturi, R. Frosting, S. Hoyer, F.
    Llinares-Lopez, F. Pedregosa and J.-P. Vert. "Efficient and Modular
    Implicit Differentiation". NeurIPS 2022"""
    name = 'jaxopt_GD'

    requirements = ["pip::jaxopt"]

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'inner_solver': ['gd', 'lbfgs'],
        'step_size_outer': [10],
        'n_inner_steps': [100],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_inner, f_outer, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):
        self.f_inner = partial(f_inner, start=0, batch_size=n_inner_samples)
        self.f_outer = partial(f_outer, start=0, batch_size=n_outer_samples)

        if self.inner_solver == 'gd':
            solver = jaxopt.GradientDescent(
                fun=self.f_inner, maxiter=self.n_inner_steps,
                implicit_diff=True, acceleration=False
            )
        elif self.inner_solver == 'lbfgs':
            solver = jaxopt.LBFGS(
                fun=self.f_inner, maxiter=self.n_inner_steps,
                implicit_diff=True,
            )
        else:
            raise ValueError(
                f"Inner solver {self.inner_solver} not available"
            )

        def value_fun(inner_var, outer_var):
            """Solver used to solve the inner problem.

            The output of this function is differentiable w.r.t. the
            outer_variable. The Jacobian is computed using implicit
            differentiation with a conjugate gradient solver.
            """
            inner_var = solver.run(inner_var, outer_var).params
            return self.f_outer(inner_var, outer_var), inner_var

        self.value_grad = jax.jit(jax.value_and_grad(
            value_fun, argnums=1, has_aux=True
        ))

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        self.run_once(2)

    def run(self, callback):

        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()

        step_sizes = jnp.array(
            [self.step_size_outer]
        )
        exponents = jnp.zeros(1)
        state_lr = init_lr_scheduler(step_sizes, exponents)

        while callback():
            outer_lr, state_lr = update_lr(state_lr)
            (_, self.inner_var), implicit_grad = self.value_grad(
                self.inner_var, self.outer_var
            )
            self.outer_var -= outer_lr * implicit_grad

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var)
