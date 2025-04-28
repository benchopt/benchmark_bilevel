import functools

from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp
    import numpy as np


class Solver(StochasticJaxSolver):
    name = 'SPABA'
    """Stochastic ProbAbilistic Bilevel Algorithm (SPABA).

    Tianshu Chu, Dachuan Xu, Wei Yao, Jin Zhang "SPABA: A Single-Loop and Probabilistic Stochastic 
    Bilevel Algorithm Achieving Optimal Sample Complexity".
    ICML 2024"""


    # any parameter defined here is accessible as a class attribute
    parameters = {
        'inner_size': [1.],
        'assis_ratio': [1.],
        'outer_ratio': [1.],
        'batch_size': [64],
        'mode_init_memory': ["zero"],
        **StochasticJaxSolver.parameters
    }

    def init(self):
        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()
        v = jnp.zeros_like(self.inner_var)

        # Init lr scheduler
        step_sizes = jnp.array(
            [self.inner_size, self.inner_size / self.outer_ratio, self.inner_size / self.assis_ratio]
        )
        # SPABA works with constant stepsizes
        exponents = jnp.zeros_like(step_sizes)
        state_lr = init_lr_scheduler(step_sizes, exponents)
        
        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var, 
            v=v,
            inner_var_old=self.inner_var.copy(),
            outer_var_old=self.outer_var.copy(),
            v_old=v.copy(),
            d_inner=jnp.zeros_like(self.inner_var),
            d_v=jnp.zeros_like(self.inner_var),
            d_outer=jnp.ones_like(self.outer_var),
            state_lr= state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )

    def get_step(self, inner_sampler, outer_sampler):

        # Gradients
        grad_inner = jax.grad(self.f_inner, argnums=0)
        grad_outer = jax.grad(self.f_outer, argnums=(0, 1))

        # Full batch gradients
        f_inner_fb = functools.partial(
            self.f_inner, start=0, batch_size=self.n_inner_samples
        )
        f_outer_fb = functools.partial(
            self.f_outer, start=0, batch_size=self.n_outer_samples
        )
        grad_inner_fb = jax.grad(f_inner_fb, argnums=0)
        grad_outer_fb = jax.grad(f_outer_fb, argnums=(0, 1))


        def spaba_one_iter(carry, _):
            (inner_lr, outer_lr, assis_lr), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            a = np.random.binomial(1, 0.001, 1)
            r = 1

            if a == 1:
        
                start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
                )
                grad_inner_var, vjp_train = jax.vjp(
                lambda z, x: grad_inner(z, x, start_inner), carry['inner_var'],
                carry['outer_var']
                )
                hvp, cross_v = vjp_train(carry['v'])

                grad_inner_var_old, vjp_train_old = jax.vjp(
                    lambda z, x: grad_inner(z, x, start_inner), carry['inner_var_old'],
                    carry['outer_var_old']
                )
                hvp_old, cross_v_old = vjp_train_old(carry['v_old'])

                start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                    carry['state_outer_sampler']
                )
                grad_in_outer, grad_out_outer = grad_outer(
                    carry['inner_var'], carry['outer_var'], start_outer
                )

                grad_in_outer_old, grad_out_outer_old = grad_outer(
                    carry['inner_var_old'], carry['outer_var_old'], start_outer
                )

                # Step.2 - Save the current variables
                carry['v_old'] = carry['v']
                carry['inner_var_old'] = carry['inner_var']
                carry['outer_var_old'] = carry['outer_var']
                
                # Step.3 - update the direction
                carry['d_v'] += (hvp - grad_in_outer) - (hvp_old - grad_in_outer_old)
                carry['d_inner'] += grad_inner_var - grad_inner_var
                carry['d_outer'] += (grad_out_outer - cross_v) - (grad_out_outer_old - cross_v_old)

                # Step.4 - update the variables
                carry['inner_var'] -= inner_lr * carry['d_inner']
                carry['v'] -= assis_lr * carry['d_v']
                carry['outer_var'] -= outer_lr * carry['d_outer']
            
            else:
                # Step.5 - get all gradients and compute the implicit gradient.(full-batch)

                grad_inner_var_fb, vjp_train_fb = jax.vjp(
                lambda z, x: grad_inner_fb(z, x), carry['inner_var'],
                carry['outer_var']
                )
                hvp_fb, cross_v_fb= vjp_train_fb(carry['v'])


                grad_in_outer_fb, grad_out_outer_fb = grad_outer_fb(
                carry['inner_var'], carry['outer_var'])
                
                carry['v_old'] = carry['v']
                carry['inner_var_old'] = carry['inner_var']
                carry['outer_var_old'] = carry['outer_var']
                
                # Step.6 - update the variables
                carry['inner_var'] -= inner_lr * grad_inner_var_fb
                carry['v'] -= assis_lr * (hvp_fb - grad_in_outer_fb)
                carry['outer_var'] -= outer_lr * (grad_out_outer_fb - cross_v_fb)
                
            carry['v'] = jnp.clip(carry['v'], -r, r)

            return carry, _

        return spaba_one_iter
