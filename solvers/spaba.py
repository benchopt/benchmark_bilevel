import functools

from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp

class Solver(StochasticJaxSolver):
    """Stochastic ProbAbilistic Bilevel Algorithm (SPABA).
    Tianshu Chu, Dachuan Xu, Wei Yao, Jin Zhang "SPABA: A Single-Loop and Probabilistic Stochastic 
    Bilevel Algorithm Achieving Optimal Sample Complexity".
    ICML 2024"""
    
    name = 'SPABA'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'inner_size': [1.],
        'assis_ratio': [1.],
        'outer_ratio': [1.],
        'batch_size': [64],
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
        exponents = jnp.zeros_like(step_sizes)
        state_lr = init_lr_scheduler(step_sizes, exponents)

        key = jax.random.PRNGKey(0)  # Initialize random key

        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var, v=v,
            inner_var_old=self.inner_var.copy(),
            outer_var_old=self.outer_var.copy(),
            v_old=v.copy(),
            d_inner=jnp.zeros_like(self.inner_var),
            d_v=jnp.zeros_like(self.inner_var),
            d_outer=jnp.zeros_like(self.outer_var),
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
            key=key  # Add key to carry
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

        def fb_directions(inner_var, outer_var, v, inner_var_old,
                          outer_var_old, v_old, d_inner, d_v, d_outer,
                          state_inner_sampler, state_outer_sampler, key):
            d_inner, vjp_train = jax.vjp(
                lambda z, x: grad_inner_fb(z, x),
                inner_var, outer_var
            )
            hvp, cross_v = vjp_train(v)
            grad_outer_in, grad_outer_out = grad_outer_fb(inner_var, outer_var)
            d_v = hvp + grad_outer_in
            d_outer = cross_v + grad_outer_out
            return (d_inner, d_v, d_outer, state_inner_sampler,
                    state_outer_sampler, key)

        def accelerate_directions(inner_var, outer_var, v, inner_var_old,
                            outer_var_old, v_old, d_inner, d_v, d_outer,
                            state_inner_sampler, state_outer_sampler, key):
            start_inner, *_, state_inner_sampler = (
                inner_sampler(state_inner_sampler))
            start_outer, *_, state_outer_sampler = (
                outer_sampler(state_outer_sampler))

            grad_inner_var, vjp_train = jax.vjp(
                lambda z, x: grad_inner(z, x, start_inner),
                inner_var, outer_var
            )
            hvp, cross_v = vjp_train(v)
            grad_outer_in, grad_outer_out = grad_outer(
                inner_var, outer_var, start_outer
            )

            grad_inner_var_old, vjp_train_old = jax.vjp(
                lambda z, x: grad_inner(z, x, start_inner),
                inner_var_old, outer_var_old
            )
            hvp_old, cross_v_old = vjp_train_old(v_old)
            grad_outer_in_old, grad_outer_out_old = grad_outer(inner_var_old,
                                                               outer_var_old,
                                                               start_outer)

            d_inner += grad_inner_var - grad_inner_var_old
            d_v += (hvp - hvp_old) + (grad_outer_in - grad_outer_in_old)
            d_outer += (cross_v - cross_v_old)
            d_outer += (grad_outer_out - grad_outer_out_old)

            return (d_inner, d_v, d_outer, state_inner_sampler,
                    state_outer_sampler, key)

        def spaba_one_iter(carry, i):
            (inner_lr, outer_lr, assis_lr), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Use JAX's random key for bernoulli distribution
            key, subkey = jax.random.split(carry['key'])  # Split the key
            i = jax.random.bernoulli(subkey, p=0.9, shape=(1,))
            i = i[0]  # Convert from array to scalar
            r = 1

            carry['d_inner'], carry['d_v'], carry['d_outer'], \
                carry['state_inner_sampler'], carry['state_outer_sampler'], \
                carry['key'] = jax.lax.cond(
                    i == 0, fb_directions, accelerate_directions,
                    carry['inner_var'], carry['outer_var'], carry['v'],
                    carry['inner_var_old'], carry['outer_var_old'],
                    carry['v_old'], carry['d_inner'], carry['d_v'],
                    carry['d_outer'], carry['state_inner_sampler'],
                    carry['state_outer_sampler'], carry['key']
                )

            carry['inner_var_old'] = carry['inner_var'].copy()
            carry['v_old'] = carry['v'].copy()
            carry['outer_var_old'] = carry['outer_var'].copy()

            # Update of the variables
            carry['inner_var'] -= inner_lr * carry['d_inner']
            carry['v'] -= assis_lr * carry['d_v']
            carry['outer_var'] -= outer_lr * carry['d_outer']
            carry['v'] = jnp.clip(carry['v'], -r, r)
            carry['key'] = key
            return carry, None

        return spaba_one_iter
