from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp


class Solver(StochasticJaxSolver):
    """Stochastic Average Bilevel Algorithm (SABA).

    M. Dagréou, P. Ablin, S. Vaiter and T. Moreau, "A framework for bilevel
    optimization that enables stochastic and global variance reduction
    algorithms", NeurIPS 2022."""
    name = 'SABA'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
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
            [self.step_size, self.step_size / self.outer_ratio]
        )
        # SABA works with constant stepsizes
        exponents = jnp.zeros_like(step_sizes)
        state_lr = init_lr_scheduler(step_sizes, exponents)

        # Initialize the memory for the variance reduction
        memory = init_memory(
            self.f_inner, self.f_outer,
            self.inner_var, self.outer_var, v,
            n_inner_samples=self.n_inner_samples,
            n_outer_samples=self.n_outer_samples,
            batch_size_inner=self.batch_size_inner,
            batch_size_outer=self.batch_size_outer,
            inner_size=self.inner_var.shape[0],
            outer_size=self.outer_var.shape[0],
        )
        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var, v=v,
            state_lr=state_lr, memory=memory,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )

    def get_step(self, inner_sampler, outer_sampler):

        grad_inner = jax.grad(self.f_inner, argnums=0)
        grad_outer = jax.grad(self.f_outer, argnums=(0, 1))

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

        def saba_one_iter(carry, _):
            (inner_step_size, outer_step_size), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Step.1 - get all gradients and compute the implicit gradient.
            (start_inner, id_inner, weight_inner,
             carry['state_inner_sampler']) = (
                inner_sampler(carry['state_inner_sampler'])
            )
            grad_inner_var, vjp_train = jax.vjp(
                lambda z, x: grad_inner(z, x, start_inner), carry['inner_var'],
                carry['outer_var']
            )
            hvp, cross_v = vjp_train(carry['v'])

            (start_outer, id_outer, weight_outer,
             carry['state_outer_sampler']) = (
                outer_sampler(carry['state_outer_sampler'])
            )
            grad_in_outer, grad_out_outer = grad_outer(
                carry['inner_var'], carry['outer_var'], start_outer
            )

            # Step 2. - update the memory and performs the variance reduction
            # here memory_*[-2] corresponds to the running average of
            # the gradients and memory[-1] to the current direction
            updates = {
                'inner_grad': (grad_inner_var, id_inner, weight_inner),
                'hvp': (hvp, id_inner, weight_inner),
                'cross_v': (cross_v, id_inner, weight_inner),
                'grad_in_outer': (grad_in_outer, id_outer, weight_outer),
                'grad_out_outer': (grad_out_outer, id_outer, weight_outer),
            }
            carry['memory'] = jax.tree_map(
                lambda mem, up: variance_reduction(mem, *up),
                carry['memory'], updates
            )

            # Step.3 - update inner variable with SGD.
            carry['inner_var'] -= inner_step_size * grad_inner_var
            carry['v'] -= inner_step_size * (hvp + grad_in_outer)
            carry['outer_var'] -= outer_step_size * (cross_v + grad_out_outer)

            return carry, _

        return saba_one_iter


def init_memory(
    inner_oracle,
    outer_oracle,
    inner_var,
    outer_var,
    v,
    n_inner_samples=1,
    n_outer_samples=1,
    batch_size_inner=1,
    batch_size_outer=1,
    inner_size=1,
    outer_size=1,
    mode="zero",
):
    n_outer = (n_outer_samples + batch_size_outer - 1) // batch_size_outer
    n_inner = (n_inner_samples + batch_size_inner - 1) // batch_size_inner
    memory = {
        'inner_grad': jnp.zeros((n_inner + 2, inner_size)),
        'hvp': jnp.zeros((n_inner + 2, inner_size)),
        'cross_v': jnp.zeros((n_inner + 2, outer_size)),
        'grad_in_outer': jnp.zeros((n_outer + 2, inner_size)),
        'grad_out_outer': jnp.zeros((n_outer + 2, outer_size)),
    }
    if mode == "full":
        grad_inner = jax.jit(jax.grad(inner_oracle, argnums=0))
        grad_outer = jax.jit(jax.grad(outer_oracle, argnums=(0, 1)))
        memory = _init_memory_fb(
            memory,
            grad_inner,
            grad_outer,
            inner_var,
            outer_var,
            v,
            n_inner_samples=n_inner_samples,
            n_outer_samples=n_outer_samples,
            batch_size_inner=batch_size_inner,
            batch_size_outer=batch_size_outer,
            inner_size=inner_size,
            outer_size=outer_size,
        )

    return memory


def _init_memory_fb(
        memory,
        grad_inner,
        grad_outer,
        inner_var,
        outer_var,
        v,
        inner_sampler,
        outer_sampler,
        state_inner_sampler,
        state_outer_sampler,
        n_inner_samples=1,
        n_outer_samples=1,
        batch_size_inner=1,
        batch_size_outer=1,
):
    n_outer = (n_outer_samples + batch_size_outer - 1) // batch_size_outer
    n_inner = (n_inner_samples + batch_size_inner - 1) // batch_size_inner
    for _ in range(n_inner):
        start_inner, state_inner_sampler = inner_sampler(state_inner_sampler)
        id_inner = state_inner_sampler['batch_order'][
            state_inner_sampler['i_batch']
        ]
        weight = batch_size_inner / n_inner_samples
        grad_inner_var, vjp_train = jax.vjp(
            lambda z, x: grad_inner(z, x, start_inner), inner_var,
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

    for id_outer in range(n_outer):
        start_outer, state_outer_sampler = outer_sampler(state_outer_sampler)
        id_outer = state_outer_sampler['batch_order'][
            state_outer_sampler['i_batch']
        ]
        weight = batch_size_outer / n_outer_samples

        grad_in, grad_out = grad_outer(inner_var, outer_var, start_outer)

        memory['grad_in_outer'] = (
            memory['grad_in_outer'].at[id_outer].set(grad_in)
            .at[-2].add(weight * grad_in)
        )
        memory['grad_out_outer'] = (
            memory['grad_out_outer'].at[id_outer].set(grad_out)
            .at[-2].add(weight * grad_out)
        )

    return memory
