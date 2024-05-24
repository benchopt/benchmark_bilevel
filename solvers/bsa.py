from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import jax
    import jax.numpy as jnp
    from functools import partial

    from benchmark_utils.sgd_inner import sgd_inner_jax
    from benchmark_utils.hessian_approximation import hia_jax
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler


class Solver(StochasticJaxSolver):
    """Bilevel Stochastic Approximation (BSA).

    S. Ghadimi and M. Wang. "Approximation Methods for Bilevel Programm".
    arxiv:1802.02246 2018"""
    name = 'BSA'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_inner_steps': [10],
        'n_hia_steps': [10],
        'batch_size': [64],
        **StochasticJaxSolver.parameters
    }

    def init(self):
        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()

        step_sizes = jnp.array(
            [self.step_size, self.step_size,
                self.step_size / self.outer_ratio]
        )
        exponents = jnp.array([.5, 0., .5])
        state_lr = init_lr_scheduler(step_sizes, exponents)
        keys = jax.random.split(jax.random.PRNGKey(self.random_state), 3)

        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var,
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
            key=keys[-1]
        )

    def get_step(self, inner_sampler, outer_sampler):
        grad_inner_fun = jax.grad(self.f_inner, argnums=0)
        grad_outer_fun = jax.grad(self.f_outer, argnums=(0, 1))

        hia = partial(
            hia_jax, grad_inner=grad_inner_fun, n_steps=self.n_hia_steps,
            sampler=inner_sampler
        )

        sgd_inner = partial(
            sgd_inner_jax, grad_inner=grad_inner_fun,
            sampler=inner_sampler, n_steps=self.n_inner_steps
        )

        def bsa_one_iter(carry, _):

            (inner_lr, hia_lr, outer_lr), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            grad_in, grad_out = grad_outer_fun(
                carry['inner_var'], carry['outer_var'], start_outer)

            implicit_grad, carry['key'], carry['state_inner_sampler'] = hia(
                carry['inner_var'], carry['outer_var'], grad_in,
                carry['state_inner_sampler'], hia_lr, key=carry['key']
            )
            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            _, vjp_fun = jax.vjp(
                lambda x: grad_inner_fun(carry['inner_var'], x, start_inner),
                carry['outer_var']
            )
            implicit_grad = vjp_fun(implicit_grad)[0]
            grad_outer_var = grad_out - implicit_grad

            carry['outer_var'] -= outer_lr * grad_outer_var
            # inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

            carry['inner_var'], carry['state_inner_sampler'] = sgd_inner(
                carry['inner_var'], carry['outer_var'],
                carry['state_inner_sampler'], step_size=inner_lr
            )

            return carry, _

        return bsa_one_iter
