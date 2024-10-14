from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.sgd_inner import sgd_inner_jax
    from benchmark_utils.hessian_approximation import sgd_v_jax
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp
    from functools import partial


class Solver(StochasticJaxSolver):
    """Amortized Implicit Gradient Optimization (AmIGO).

    M. Arbel and J. Mairal. "Amortized Implicit Differentiation for Stochastic
    Bilevel Optimization". ICLR 2022"""
    name = 'AmIGO'

    """How to add a new stochastic solver to the benchmark?

    Stochastic solvers are Solver classes that inherit from the
    `StochasticJaxSolver` class. They should implement the `init` and the
    `get_step_methods` and the class variable `parameters`.

    * The variable `parameters` is a dictionary that contains the solver's
    parameters. In the case of AmIGO, it contains
        - step_size: the step_size of the inner and linear system solvers
        - outer_ratio: the ratio between the step sizes of the inner and the
        outer updates
        - n_inner_steps: the number of steps of the inner and the linear system
        solvers
        - batch_size: the size of the minibatch (assumed to be the same for the
        inner and outer functions)
        - **StochasticJaxSolver.parameters: the parameters shared by all the
        stochastic solvers based on the StochasticJaxSolver class

    * The `init` methods initializes variables that are udapted during the
    optimization process. In the case of AmIGO, it initializes the inner and
    outer variables, the linear system variable v and the learning rate
    scheduler. It returns a dictionary containing these variables and the
    initial state of the samplers. Those ones are already provided by the
    attributes `state_inner_sampler` and `state_outer_sampler`.

    * The `get_step` method returns a function that performs one iteration of
    the optimization algorithm. This function should be jittable by JAX. In
    this function are also initialized the eventual subroutines such as the
    inner SGD and the linear system solver in the case of AmIGO. Note that the
    variable updated during the process are stored in the `carry` dictionary,
    whose initial state is the output of the `init` method.
    """

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_inner_steps': [10],
        'batch_size': [64],
        **StochasticJaxSolver.parameters
        # Contains the `eval_freq` and `random_state`parameters
    }

    def init(self):

        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()

        v = jnp.zeros_like(self.inner_var)

        step_sizes = jnp.array(
            [self.step_size, self.step_size,
                self.step_size / self.outer_ratio]
        )
        exponents = jnp.zeros(3)
        state_lr = init_lr_scheduler(step_sizes, exponents)

        # The return dictionary should contain all the variables that are
        # updated during the optimization process. The state of the samplers
        # are already provided by the attributes `state_inner_sampler` and
        # `state_outer_sampler`.
        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var, v=v,
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )

    def get_step(self, inner_sampler, outer_sampler):
        grad_inner_fun = jax.grad(self.f_inner, argnums=0)
        grad_outer_fun = jax.grad(self.f_outer, argnums=(0, 1))

        sgd_inner = partial(
            sgd_inner_jax, grad_inner=grad_inner_fun,
            sampler=inner_sampler, n_steps=self.n_inner_steps
        )
        sgd_v = partial(
            sgd_v_jax, grad_inner=grad_inner_fun,
            sampler=inner_sampler, n_steps=self.n_inner_steps
        )

        # This function should be jittable by JAX. It returns the output of
        # one iteration of the optimization algorithm  (one iteration = one
        # outer vairable update in this case).
        def amigo_one_iter(carry, _):

            (inner_lr, v_lr, outer_lr), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Update the inner_var
            carry['inner_var'], carry['state_inner_sampler'] = sgd_inner(
                carry['inner_var'], carry['outer_var'],
                carry['state_inner_sampler'], step_size=inner_lr
            )

            # Get outer gradient
            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            grad_in, grad_out = grad_outer_fun(
                carry['inner_var'], carry['outer_var'], start_outer
            )

            # compute SGD for the auxillary variable
            carry['v'], carry['state_inner_sampler'] = sgd_v(
                carry['inner_var'], carry['outer_var'], carry['v'], grad_in,
                carry['state_inner_sampler'], v_lr, sampler=inner_sampler,
            )
            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            _, vjp_fun = jax.vjp(
                lambda x: grad_inner_fun(carry['inner_var'], x, start_inner),
                carry['outer_var']
            )
            implicit_grad = vjp_fun(carry['v'])[0]
            grad_outer_var = grad_out - implicit_grad

            # Update the outer variable
            carry['outer_var'] -= outer_lr * grad_outer_var

            return carry, _

        return amigo_one_iter
