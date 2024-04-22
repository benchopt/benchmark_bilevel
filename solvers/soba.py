from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.get_memory import get_memory
    from benchmark_utils.minibatch_sampler import init_sampler
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec
    from benchmark_utils.oracles import MultiLogRegOracle, DataCleaningOracle
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler

    import jax
    import jax.numpy as jnp
    from functools import partial


class Solver(BaseSolver):
    """Stochastic Bilevel Algorithm (SOBA).

    M. Dagréou, P. Ablin, S. Vaiter and T. Moreau, "A framework for bilevel
    optimization that enables stochastic and global variance reduction
    algorithms", NeurIPS 2022."""
    name = 'SOBA'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'batch_size': [64],
        'eval_freq': [128],
        'random_state': [1],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_inner, f_outer, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):
        """Set the problem to solve.

        Parameters
        ----------
        f_inner, f_outer: callable
            Inner and outer objective function for the bilevel optimization
            problem. Should take in
        """

        self.f_inner = f_inner
        self.f_outer = f_outer
        self.n_inner_samples = n_inner_samples
        self.n_outer_samples = n_outer_samples

        if self.batch_size == "full":
            self.batch_size_inner = n_inner_samples
            self.batch_size_outer = n_outer_samples
        else:
            self.batch_size_inner = self.batch_size
            self.batch_size_outer = self.batch_size

        self.f_inner = partial(self.f_inner, batch_size=self.batch_size_inner)
        self.f_outer = partial(self.f_outer, batch_size=self.batch_size_outer)

        inner_sampler, self.state_inner_sampler = init_sampler(
            n_samples=n_inner_samples, batch_size=self.batch_size_inner
        )
        outer_sampler, self.state_outer_sampler = init_sampler(
            n_samples=n_outer_samples, batch_size=self.batch_size_outer
        )
        self.one_epoch = self.get_one_epoch_jitted(
            inner_sampler, outer_sampler
        )

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        # warmup
        self.run_once(2)

    def init(self):
        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()

        v = jnp.zeros_like(inner_var)
        # Init lr scheduler
        step_sizes = jnp.array(
            [self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = jnp.array(
            [.5, .5]
        )
        state_lr = init_lr_scheduler(step_sizes, exponents)
        return dict(
            inner_var=inner_var, outer_var=outer_var, v=v,
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )


    def run(self, callback):
        eval_freq = self.eval_freq
        memory_start = get_memory()

        carry = self.init()

        # Start algorithm
        while callback():
            carry = self.one_epoch(carry, self.eval_freq)
            self.inner_var = carry["inner_var"],
            self.outer_var = carry["outer_var"]

    def get_result(self):
        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var
        )

    def get_step(self, inner_sampler, outer_sampler):

        grad_inner = jax.grad(self.f_inner, argnums=0)
        grad_outer = jax.grad(self.f_outer, argnums=(0, 1))

        def soba_one_iter(carry, _):

            (inner_step_size, outer_step_size), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Step.1 - get all gradients and compute the implicit gradient.
            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            grad_inner_var, vjp_train = jax.vjp(
                lambda z, x: grad_inner(z, x, start_inner), carry['inner_var'],
                carry['outer_var']
            )
            hvp, cross_v = vjp_train(carry['v'])

            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            grad_in_outer, grad_out_outer = grad_outer(
                carry['inner_var'], carry['outer_var'], start_outer
            )

            # Step.2 - update inner variable with SGD.
            carry['inner_var'] -= inner_step_size * grad_inner_var
            carry['v'] -= inner_step_size * (hvp + grad_in_outer)
            carry['outer_var'] -= outer_step_size * (cross_v + grad_out_outer)

            return carry, _

        return soba_one_iter

    def get_one_epoch_jitted(self, inner_sampler, outer_sampler):
        step = self.get_step(inner_sampler, outer_sampler)

        def one_epoch(carry, eval_freq):
            carry, _ = jax.lax.scan(
                step, init=carry, xs=None,
                length=eval_freq,
            )
            return carry

        return jax.jit(
            one_epoch, static_argnums=(1,)
        )
