from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver


from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import jax
    import jax.numpy as jnp
    from functools import partial

    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.hessian_approximation import joint_shia_jax
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler


class Solver(StochasticJaxSolver):
    """Momentum-based Recursive Bilevel Optimizer (MRBO).

    J. Yang, K. Ji, Y. Liang. "Provabily Faster Algorithms for Bilevel
    Optimization". NeurIPS 2021"""
    name = 'MRBO'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_shia_steps': [10],
        'batch_size': [64],
        'eta': [.5],
        **StochasticJaxSolver.parameters
    }

    def init(self):
        # Init variables
        inner_var = self.inner_var.copy()
        outer_var = self.outer_var.copy()

        memory_inner = jnp.zeros((2, *inner_var.shape))
        memory_outer = jnp.zeros((2, *outer_var.shape))
        step_sizes = jnp.array(  # (inner_ss, hia_lr, eta, outer_ss)
            [
                self.step_size,
                self.step_size,
                self.eta,
                self.step_size / self.outer_ratio,
            ]
        )
        exponents = jnp.array([1/3, 0, 2/3, 1/3])
        state_lr = init_lr_scheduler(step_sizes, exponents)

        return dict(
            inner_var=inner_var, outer_var=outer_var,
            memory_inner=memory_inner, memory_outer=memory_outer,
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )

    def get_step(self, inner_sampler, outer_sampler):
        grad_inner_fun = jax.grad(self.f_inner, argnums=0)
        grad_outer_fun = jax.grad(self.f_outer, argnums=(0, 1))

        joint_shia = partial(
            joint_shia_jax, grad_outer=grad_outer_fun,
            grad_inner=grad_inner_fun, n_steps=self.n_shia_steps,
            sampler=inner_sampler
        )

        def mrbo_one_iter(carry, _):

            (inner_lr, hia_lr, eta, outer_lr), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Step.1 - Update direction for z with momentum
            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            grad_inner_var, vjp_fun = jax.vjp(
                lambda x: grad_inner_fun(carry['inner_var'], x, start_inner),
                carry['outer_var']
            )
            grad_inner_var_old, vjp_fun_old = jax.vjp(
                lambda x: grad_inner_fun(carry['memory_inner'][0], x,
                                         start_inner),
                carry['memory_outer'][0]
            )

            carry['memory_inner'] = carry['memory_inner'].at[1].set(
                grad_inner_var
                + (1-eta) * (carry['memory_inner'][1] - grad_inner_var_old)
            )

            # Step.2 - Compute implicit grad approximation with HIA
            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            grad_outer, impl_grad = grad_outer_fun(
                carry['inner_var'], carry['outer_var'], start_outer
            )
            grad_outer_old, impl_grad_old = grad_outer_fun(
                carry['memory_inner'][0], carry['memory_outer'][0], start_outer
            )

            ihvp, ihvp_old, carry['state_inner_sampler'] = joint_shia(
                carry['inner_var'], carry['outer_var'], grad_outer,
                carry['memory_inner'][0], carry['memory_outer'][0],
                grad_outer_old, carry['state_inner_sampler'], hia_lr
            )
            impl_grad -= vjp_fun(ihvp)[0]
            impl_grad_old -= vjp_fun_old(ihvp_old)[0]

            # Step.3 - Update direction for x with momentum
            carry['memory_outer'] = carry['memory_outer'].at[1].set(
                eta * impl_grad
                + (1-eta) * (carry['memory_outer'][1]
                             + impl_grad - impl_grad_old)
            )

            # Step.4 - Save the current variables
            carry['memory_inner'] = carry['memory_inner'].at[0].set(
                carry['inner_var']
            )
            carry['memory_outer'] = carry['memory_outer'].at[0].set(
                carry['outer_var']
            )

            # Step.5 - update the variables with the directions
            carry['inner_var'] -= inner_lr * carry['memory_inner'][1]
            carry['outer_var'] -= outer_lr * carry['memory_outer'][1]

            return carry, _
        return mrbo_one_iter
