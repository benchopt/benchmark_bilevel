from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils import constants
    from benchmark_utils.get_memory import get_memory
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp
    from functools import partial

    import jaxopt


class Solver(BaseSolver):
    """Iterate Differentiation with JAXopt solvers.

    M. Blondel, Q. Berthet, M. Cuturi, R. Frosting, S. Hoyer, F.
    Llinares-Lopez, F. Pedregosa and J.-P. Vert. "Efficient and Modular
    Implicit Differentiation". NeurIPS 2022"""
    name = 'jaxopt ITD'

    requirements = ["pip:jaxopt"]

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'inner_solver': ['gd'],
        'step_size_outer': [.01],
        'eval_freq': [1],
        'n_inner_steps': [100],
        'warm_start': [True, False]
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_train, f_val, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):
        self.n_inner_samples = n_inner_samples
        self.n_outer_samples = n_outer_samples

        self.f_inner = jax.jit(
            partial(f_train(framework='jax'),
                    batch_size=n_inner_samples, start=0),
        )
        self.f_outer = jax.jit(
            partial(f_val(framework='jax'),
                    batch_size=n_outer_samples, start=0),
        )

        @partial(jax.jit, static_argnames=("f", "n_steps"))
        def inner_solver_fun(outer_var, inner_var, f=None, n_steps=1):
            """Solver used to solve the inner problem.

            The output of this function is differentiable w.r.t. the
            outer_variable. The Jacobian is computed using iterative
            differentiation.
            """
            if self.inner_solver == 'gd':
                solver = jaxopt.GradientDescent(
                    fun=f, maxiter=n_steps, implicit_diff=False,
                    acceleration=False, unroll=True
                )
            else:
                raise ValueError(f"Inner solver {self.inner_solver} not"
                                 + "available")
            return solver.run(inner_var, outer_var).params

        self.inner_solver_fun = partial(inner_solver_fun,
                                        f=self.f_inner,
                                        n_steps=self.n_inner_steps)

        self.inner_var = inner_var0
        self.outer_var = outer_var0
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

    def warm_up(self):
        self.run_once(2)
        self.inner_var = self.inner_var0
        self.outer_var = self.outer_var0

    def run(self, callback):
        eval_freq = self.eval_freq

        # Init variables
        memory_start = get_memory()
        inner_var = self.inner_var.copy()
        outer_var = self.outer_var.copy()

        step_sizes = jnp.array(
            [self.step_size_outer]
        )
        exponents = jnp.zeros(1)
        state_lr = init_lr_scheduler(step_sizes, exponents)

        grad_outer = jax.jit(jax.grad(self.f_outer, argnums=(0, 1)))

        while callback():
            for _ in range(eval_freq):
                outer_lr, state_lr = update_lr(state_lr)
                init_inner = inner_var if self.warm_start else self.inner_var0
                inner_var, jvp_fun = jax.vjp(self.inner_solver_fun,
                                             outer_var,
                                             init_inner)

                grad_outer_in, grad_outer_out = grad_outer(inner_var,
                                                           outer_var)

                implicit_grad = grad_outer_out + jvp_fun(grad_outer_in)[0]
                outer_var -= outer_lr * implicit_grad
            memory_end = get_memory()
            self.inner_var = inner_var
            self.outer_var = outer_var
            self.memory = memory_end - memory_start
            self.memory /= 1e6

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var,
                    memory=self.memory)
