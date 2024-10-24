from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    # import your reusable functions here
    from benchmark_utils import constants

    # step size scheduler
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp
    from functools import partial

    import jaxopt


class Solver(BaseSolver):
    """How to add a new  solver to the benchmark?

    This template solver is an adaptation of the solver from the benchopt
    template benchmark (https://github.com/benchopt/template_benchmark/) to
    the bilevel setting. Other explanations can be found in
    https://benchopt.github.io/tutorials/add_solver.html.
    """
    # Name to select the solver in the CLI and to display the results.
    name = 'Template solver'

    # List of packages needed to run the solver.
    requirements = ["pip:jaxopt"]

    # Stopping criterion for the solver.
    # See https://benchopt.github.io/user_guide/performance_curves.html for
    # more information on benchopt stopping criteria.
    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'step_size_outer': [10],
        'n_inner_steps': [100],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_inner, f_outer, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. For the bilevel benchmark, these
        # informations are the inner and outer objective functions, the number
        # of samples to draw for the inner and outer objective functions, the
        # initial values of the inner and outer variables.
        self.f_inner = partial(f_inner, start=0, batch_size=n_inner_samples)
        self.f_outer = partial(f_outer, start=0, batch_size=n_outer_samples)
        inner_solver = jaxopt.GradientDescent(
                fun=self.f_inner, maxiter=self.n_inner_steps,
                implicit_diff=True, acceleration=False
        )

        # The value function is defined for this specific solver, but it is
        # not mandatory in general.
        def value_fun(inner_var, outer_var):
            """Value function for the bilevel optimization problem.

            The output of this function is differentiable w.r.t. the
            outer_variable. The Jacobian is computed using implicit
            differentiation with a conjugate gradient solver.
            """
            inner_var = inner_solver.run(inner_var, outer_var).params
            return self.f_outer(inner_var, outer_var), inner_var

        # The value function and its gradient are jitted by JAX for fast oracle
        # computations.
        self.value_grad = jax.jit(jax.value_and_grad(
            value_fun, argnums=1, has_aux=True
        ))

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        # Run the solver for 2 iterations for the JAX compilation if
        # applicable.
        self.run_once(2)

    def run(self, callback):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()

        step_sizes = jnp.array(
            [self.step_size_outer]
        )
        exponents = jnp.zeros(1)
        # The step size scheduler provides step sizes that have the form
        # `a / t**b`, where t is the iteration number, `a` is a constant
        # that comes from the first argument of init_lr_scheduler and `b` is
        # the exponent that comes from the second argument of
        # init_lr_scheduler.
        state_lr = init_lr_scheduler(step_sizes, exponents)

        # The function `callback` calls `Objective.get_result` to check if the
        # stopping criterion is met. More informations on the callback
        # asmpling strategy can be found here:
        # https://benchopt.github.io/user_guide/performance_curves.html#using-a
        # -callback
        while callback():
            # update_lr provides a step_size and a new state of the scheduler.
            outer_lr, state_lr = update_lr(state_lr)

            # Compute the value and the gradient of the value function. Also
            # provide the inner solution.
            (_, self.inner_var), implicit_grad = self.value_grad(
                self.inner_var, self.outer_var
            )

            # Update the outer variable by a gradient step.
            self.outer_var -= outer_lr * implicit_grad

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(inner_var=self.inner_var, outer_var=self.outer_var)
