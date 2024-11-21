from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp


class Solver(StochasticJaxSolver):
    # The docstring should contain the solver's name and a reference to the
    # paper where it is introduced. This will be displayed in the HTML result
    # page.
    """Stochastic Bilevel Algorithm (SOBA).

    M. Dagréou, P. Ablin, S. Vaiter and T. Moreau, "A framework for bilevel
    optimization that enables stochastic and global variance reduction
    algorithms", NeurIPS 2022."""
    # Name to select the solver in the CLI and to display the results.
    name = 'Template Stochastic Solver'

    """How to add a new stochastic solver to the benchmark?

    Stochastic solvers are Solver classes that inherit from the
    `StochasticJaxSolver` class which aims at making easy the JIT compilation
    of the code. Stochastic solvers should implement the `init` and the
    `get_step_methods` and the class variable `parameters`. One epoch of
    `StochasticJaxSolver` corresponds to `eval_freq` outer iterations of the
    solver. The epochs of these solvers are jitted by JAX to get fast
    stochastic iterations.

    * The variable `parameters` is a dictionary that contains the solver's
    parameters. Here, it contains
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
    optimization process. Here, it initializes the inner and
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

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'batch_size': [64],
        **StochasticJaxSolver.parameters
    }

    def init(self):
        """
        Initializes the stochastic solver.
        It returns a dictionary which is the initial state of the `carry`
        dictionary in the `get_step` method. It contains at least `inner_var`
        and `outer_var`. In this specific case, it also contains the initial
        value of the linear system variable `v`, the initial state of the
        lr scheduler, and the initial state of the samplers.
        """
    
        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()
        v = jnp.zeros_like(self.inner_var)

        # Init lr scheduler
        step_sizes = jnp.array(
            [self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = jnp.array(
            [.5, .5]
        )

        # The step size scheduler provides step sizes that have the form
        # `a / t**b`, where t is the iteration number, `a` is a constant
        # that comes from the first argument of init_lr_scheduler and `b` is
        # the exponent that comes from the second argument of
        # init_lr_scheduler.
        state_lr = init_lr_scheduler(step_sizes, exponents)
        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var, v=v,
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )

    def get_step(self, inner_sampler, outer_sampler):

        grad_inner = jax.grad(self.f_inner, argnums=0)
        grad_outer = jax.grad(self.f_outer, argnums=(0, 1))

        def soba_one_iter(carry, _):

            (inner_step_size, outer_step_size), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Step.1 - get all gradients and compute the implicit gradient.

            # First index for the samples of the inner function
            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )

            # The gradient of the inner function w.r.t. the inner variable
            # and a function that takes as input a vector v and returns
            # the different Hessian-vector products with v. (c.f. l.128)
            grad_inner_var, vjp_train = jax.vjp(
                lambda z, x: grad_inner(z, x, start_inner), carry['inner_var'],
                carry['outer_var']
            )

            # Product between the Hessian of the inner function w.r.t. the
            # inner variable and the vector v and product between the cross
            # derivatives matrix and the vector v.
            hvp, cross_v = vjp_train(carry['v'])

            # First index for the samples of the outer function
            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )

            # Gradient of the outer function
            grad_in_outer, grad_out_outer = grad_outer(
                carry['inner_var'], carry['outer_var'], start_outer
            )

            # Step.2 - update inner variable with SGD.
            # Gradient step for the inner variable step
            carry['inner_var'] -= inner_step_size * grad_inner_var
            # Gradient step for the linear system variable
            carry['v'] -= inner_step_size * (hvp + grad_in_outer)
            # Approximate gradient step for the outer variable
            carry['outer_var'] -= outer_step_size * (cross_v + grad_out_outer)

            return carry, _

        return soba_one_iter
