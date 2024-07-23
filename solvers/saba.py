from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.tree_utils import tree_scalar_mult
    from benchmark_utils.tree_utils import init_memory_of_trees
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.tree_utils import select_memory, update_memory
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler
    from benchmark_utils.tree_utils import update_sgd_fn, tree_add, tree_diff

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
        v = jax.tree_util.tree_map(jnp.zeros_like, self.inner_var)

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
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
            batch_size_inner=self.batch_size_inner,
            batch_size_outer=self.batch_size_outer,
        )
        return memory, dict(
            inner_var=self.inner_var, outer_var=self.outer_var, v=v,
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )

    def get_step(self, inner_sampler, outer_sampler):

        grad_inner = jax.grad(self.f_inner, argnums=0)
        grad_outer = jax.grad(self.f_outer, argnums=(0, 1))

        def saba_one_iter(carry, _):
            memory, carry = carry
            (inner_lr, outer_lr), carry['state_lr'] = update_lr(
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
            memory = jax.tree_util.tree_map(
                lambda mem, up: variance_reduction(mem, *up),
                memory, updates, is_leaf=lambda x: isinstance(x, tuple)
            )

            # Step.3 - update inner variable with SGD.
            carry['inner_var'] = update_sgd_fn(
                carry['inner_var'], select_memory(memory['inner_grad'], -1),
                inner_lr
            )
            carry['v'] = update_sgd_fn(
                carry['v'],
                tree_add(select_memory(memory['hvp'], -1),
                         select_memory(memory['grad_in_outer'], -1)),
                inner_lr
            )
            carry['outer_var'] = update_sgd_fn(
                carry['outer_var'],
                tree_add(select_memory(memory['cross_v'], -1),
                         select_memory(memory['grad_out_outer'], -1)),
                outer_lr
            )

            return (memory, carry), _

        return saba_one_iter

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
    mode="zero",
):
    n_batchs_outer = len(state_outer_sampler['batch_order'])
    n_batchs_inner = len(state_inner_sampler['batch_order'])
    memory = {
        'inner_grad': init_memory_of_trees(n_batchs_inner + 2, inner_var),
        'hvp': init_memory_of_trees(n_batchs_inner + 2, inner_var),
        'cross_v': init_memory_of_trees(n_batchs_inner + 2, outer_var),
        'grad_in_outer': init_memory_of_trees(n_batchs_outer + 2, inner_var),
        'grad_out_outer': init_memory_of_trees(n_batchs_outer + 2, outer_var),
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
        memory['inner_grad'] = update_memory(
            memory['inner_grad'], id_inner, grad_inner_var)
        memory['inner_grad'] = update_memory(
            memory['inner_grad'], -2,
            tree_add(select_memory(memory['inner_grad'], -2),
                     tree_scalar_mult(weight, grad_inner_var))
        )
        memory['hvp'] = update_memory(memory['hvp'], id_inner, hvp)
        memory['hvp'] = update_memory(
            memory['hvp'], -2,
            tree_add(select_memory(memory['hvp'], -2),
                     tree_scalar_mult(weight, hvp))
        )
        memory['cross_v'] = update_memory(memory['cross_v'], id_inner, cross_v)
        memory['cross_v'] = update_memory(
            memory['cross_v'], -2,
            tree_add(select_memory(memory['cross_v'], -2),
                     tree_scalar_mult(weight, cross_v))
        )

    for id_outer in range(n_batchs_outer):
        weight = batch_size_outer / n_outer_samples

        grad_in, grad_out = grad_outer(inner_var, outer_var,
                                       id_outer*batch_size_outer)

        memory['grad_in_outer'] = update_memory(memory['grad_in_outer'],
                                                id_outer, grad_in)
        memory['grad_in_outer'] = update_memory(
            memory['grad_in_outer'], -2,
            tree_add(
                select_memory(memory['grad_in_outer'], -2),
                tree_scalar_mult(weight, grad_in))
        )
        memory['grad_out_outer'] = update_memory(memory['grad_out_outer'],
                                                 id_outer, grad_out)
        memory['grad_out_outer'] = update_memory(
            memory['grad_out_outer'], -2,
            tree_add(
                select_memory(memory['grad_out_outer'], -2),
                tree_scalar_mult(weight, grad_out))
        )

    return memory


def variance_reduction(memory, grad, idx, weigth):
    diff = tree_diff(grad, memory[idx])
    direction = tree_add(diff, select_memory(memory, -2))
    memory = update_memory(memory, -1, direction)
    memory = update_memory(memory, -2, tree_scalar_mult(weigth, diff))
    memory = update_memory(memory, idx, grad)
    return memory
