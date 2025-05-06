import functools

from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp

class Solver(StochasticJaxSolver):
    """Multiple Stochastic Estimators Bilevel Algorit (MSEBA).
    Tianshu Chu, Dachuan Xu, Wei Yao, Chengming Yu, Jin Zhang "A Provably Convergent 
    Plug-and-Play Framework for Stochastic Bilevel Optimization".
    arXiv:2505.01258"""
    
    name = 'MSEBA'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
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
            [self.step_size,  self.inner_size, self.inner_size / self.outer_ratio, self.inner_size / self.assis_ratio]
        )
        exponents = jnp.zeros_like(step_sizes)
        state_lr = init_lr_scheduler(step_sizes, exponents)
        
        # Initialize the memory for the variance reduction
        memory = init_memory(
            self.f_inner, self.f_outer,
            self.inner_var, self.outer_var, v,
            n_inner_samples=self.n_inner_samples,
            n_outer_samples=self.n_outer_samples,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
            batch_size_inner=self.batch_size_inner,
            batch_size_outer=self.batch_size_outer,
            inner_size=self.inner_var.shape[0],
            outer_size=self.outer_var.shape[0],
        )

        # Initialize random key
        key = jax.random.PRNGKey(0)  

        return memory,dict(
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
                          outer_var_old, v_old,  d_v, d_outer,
                          state_inner_sampler, state_outer_sampler, key):
            
            _, vjp_train = jax.vjp(
                lambda z, x: grad_inner_fb(z, x),
                inner_var, outer_var
            )

            hvp, cross_v = vjp_train(v)
            grad_outer_in, grad_outer_out = grad_outer_fb(inner_var, outer_var)
            d_v = hvp + grad_outer_in
            d_outer = cross_v + grad_outer_out
            return (d_v, d_outer, state_inner_sampler,
                    state_outer_sampler, key)

        def accelerate_directions(inner_var, outer_var, v, inner_var_old,
                            outer_var_old, v_old, d_v, d_outer,
                            state_inner_sampler, state_outer_sampler, key):
            start_inner, *_, state_inner_sampler = (
                inner_sampler(state_inner_sampler))
            start_outer, *_, state_outer_sampler = (
                outer_sampler(state_outer_sampler))
            
            _, vjp_train = jax.vjp(
                lambda z, x: grad_inner(z, x, start_inner),
                inner_var, outer_var
            )

            hvp, cross_v = vjp_train(v)
            grad_outer_in, grad_outer_out = grad_outer(
                inner_var, outer_var, start_outer
            )

            _, vjp_train_old = jax.vjp(
                lambda z, x: grad_inner(z, x, start_inner),
                inner_var_old, outer_var_old
            )
            hvp_old, cross_v_old = vjp_train_old(v_old)
            grad_outer_in_old, grad_outer_out_old = grad_outer(inner_var_old,
                                                               outer_var_old,
                                                               start_outer)

            d_v += (hvp - hvp_old) + (grad_outer_in - grad_outer_in_old)
            d_outer += (cross_v - cross_v_old)
            d_outer += (grad_outer_out - grad_outer_out_old)

            return (d_v, d_outer, state_inner_sampler,
                    state_outer_sampler, key)

        def mseba_one_iter(carry, _):
            memory, carry = carry
            (step_lr, inner_lr, outer_lr, assis_lr), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Use JAX's random key for bernoulli distribution
            key, subkey = jax.random.split(carry['key'])  # Split the key
            i = jax.random.bernoulli(subkey, p=0.95, shape=(1,))
            i = i[0]  # Convert from array to scalar
            r = 1

            (start_inner, id_inner, weight_inner,
             carry['state_inner_sampler']) = (
                inner_sampler(carry['state_inner_sampler'])
            )
            # previous gradient
            grad_inner_var_old = grad_inner(carry['inner_var_old'], carry['outer_var_old'], start_inner)

            grad_inner_var = grad_inner(carry['inner_var'], carry['outer_var'], start_inner)

            updates = {
                'inner_grad': (grad_inner_var, id_inner, weight_inner),
            }
            memory = {
                **memory,
                'inner_grad': variance_reduction(memory['inner_grad'], *updates['inner_grad']),
            }

            carry['d_v'], carry['d_outer'], \
                carry['state_inner_sampler'], carry['state_outer_sampler'], \
                carry['key'] = jax.lax.cond(
                    i == 0, fb_directions, accelerate_directions,
                    carry['inner_var'], carry['outer_var'], carry['v'],
                    carry['inner_var_old'], carry['outer_var_old'],
                    carry['v_old'], carry['d_v'],carry['d_outer'], 
                    carry['state_inner_sampler'],
                    carry['state_outer_sampler'], carry['key']
                )

            carry['inner_var_old'] = carry['inner_var'].copy()
            carry['v_old'] = carry['v'].copy()
            carry['outer_var_old'] = carry['outer_var'].copy()
            carry['d_inner'] = (1 - step_lr) * (carry['d_inner'] - grad_inner_var_old) + (1 + step_lr) * grad_inner_var  + step_lr * (memory['inner_grad'][-1] )

            # Update of the variables
            carry['inner_var'] -= inner_lr * carry['d_inner']
            carry['v'] -= assis_lr * carry['d_v']
            carry['outer_var'] -= outer_lr * carry['d_outer']
            carry['v'] = jnp.clip(carry['v'], -r, r)
            carry['key'] = key
            return (memory, carry), _

        return mseba_one_iter
    
    def get_one_epoch_jitted(self, inner_sampler, outer_sampler):
        step = self.get_step(inner_sampler, outer_sampler)

        def one_epoch(carry, memory, eval_freq):
            (memory, carry), _ = jax.lax.scan(
                step, init=(memory, carry), xs=None,
                length=eval_freq,
            )
            return memory, carry

        return jax.jit(
            one_epoch, static_argnums=2,
            donate_argnums=1
        )

    def run(self, callback):
        memory, carry = self.init()

        # Start algorithm
        while callback():
            memory, carry = self.one_epoch(carry, memory, self.eval_freq)
            self.inner_var = carry["inner_var"]
            self.outer_var = carry["outer_var"]
    
def init_memory(
    f_inner,
    f_outer,
    inner_var,
    outer_var,
    v,
    n_inner_samples=1,
    n_outer_samples=1,
    batch_size_inner=1,
    batch_size_outer=1,
    state_inner_sampler=None,
    state_outer_sampler=None,
    inner_size=1,
    outer_size=1,
    mode="zero",
):
    n_batchs_outer = len(state_outer_sampler['batch_order'])
    n_batchs_inner = len(state_inner_sampler['batch_order'])
    memory = {
        'inner_grad': jnp.zeros((n_batchs_inner + 2, inner_size)),
        'hvp': jnp.zeros((n_batchs_inner + 2, inner_size)),
        'cross_v': jnp.zeros((n_batchs_inner + 2, outer_size)),
        'grad_in_outer': jnp.zeros((n_batchs_outer + 2, inner_size)),
        'grad_out_outer': jnp.zeros((n_batchs_outer + 2, outer_size)),
    }
    if mode == "full":
        grad_inner = jax.jit(jax.grad(f_inner, argnums=0))
        grad_outer = jax.jit(jax.grad(f_outer, argnums=(0, 1)))
        memory = _init_memory_fb(
            memory,
            grad_inner,
            grad_outer,
            inner_var,
            outer_var,
            v,
            n_batchs_inner=n_batchs_inner,
            n_batchs_outer=n_batchs_outer,
            n_inner_samples=n_inner_samples,
            n_outer_samples=n_outer_samples,
            batch_size_inner=batch_size_inner,
            batch_size_outer=batch_size_outer,
        )

    return memory


def _init_memory_fb(
        memory,
        grad_inner,
        grad_outer,
        inner_var,
        outer_var,
        v,
        n_batchs_inner=1,
        n_batchs_outer=1,
        n_inner_samples=1,
        n_outer_samples=1,
        batch_size_inner=1,
        batch_size_outer=1,
):
    for id_inner in range(n_batchs_inner):
        weight = batch_size_inner / n_inner_samples
        grad_inner_var, vjp_train = jax.vjp(
            lambda z, x: grad_inner(z, x, id_inner*batch_size_inner),
            inner_var,
            outer_var
        )
        hvp, cross_v = vjp_train(v)
        memory['inner_grad'] = (
            memory['inner_grad'].at[id_inner].set(grad_inner_var)
            .at[-2].add(weight * grad_inner_var)
        )
        memory['hvp'] = (
            memory['hvp'].at[id_inner].set(hvp)
            .at[-2].add(weight * hvp)
        )
        memory['cross_v'] = (
            memory['cross_v'].at[id_inner].set(cross_v)
            .at[-2].add(weight * cross_v)
        )

    for id_outer in range(n_batchs_outer):
        weight = batch_size_outer / n_outer_samples

        grad_in, grad_out = grad_outer(inner_var, outer_var,
                                       id_outer*batch_size_outer)

        memory['grad_in_outer'] = (
            memory['grad_in_outer'].at[id_outer].set(grad_in)
            .at[-2].add(weight * grad_in)
        )
        memory['grad_out_outer'] = (
            memory['grad_out_outer'].at[id_outer].set(grad_out)
            .at[-2].add(weight * grad_out)
        )

    return memory

def variance_reduction(memory, grad, idx, weigth):
    diff = grad - memory[idx]
    direction = diff + memory[-2]
    memory = (
        memory
        .at[-1].set(direction)
        .at[-2].add(weigth * diff)
        .at[idx].set(grad)
    )
    return memory
