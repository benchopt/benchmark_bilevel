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
    """Fully First-order Stochastic Approximation (F2SA).

    J. Kwon, D. Kwon, S. Wright and R. Noewak, "A Fully First-Order Method for
    Stochastic Bilevel Optimization", ICML 2023."""
    name = 'F2SA'

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
        'framework': ["jax"],
        'lmbda0': [1.],
        'delta_lmbda': [.1],
        'n_inner_steps': [10],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def skip(self, f_train, f_val, **kwargs):
        if self.framework == 'numba':
            if self.batch_size == 'full':
                return True, "Numba is not useful for full bach resolution."
            elif isinstance(f_train(),
                            (MultiLogRegOracle, DataCleaningOracle)):
                return True, "Numba implementation not available for " \
                      "this oracle."
            elif isinstance(f_val(), (MultiLogRegOracle, DataCleaningOracle)):
                return True, "Numba implementation not available for" \
                      "this oracle."
        elif self.framework not in ['none', 'numba']:
            return True, f"Framework {self.framework} not supported."

        try:
            f_train(framework=self.framework)
        except NotImplementedError:
            return (
                True,
                f"Framework {self.framework} not compatible with "
                f"oracle {f_train()}"
            )
        return False, None

    def set_objective(self, f_train, f_val, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):

        self.f_inner = f_train(framework=self.framework)
        self.f_outer = f_val(framework=self.framework)
        self.n_inner_samples = n_inner_samples
        self.n_outer_samples = n_outer_samples

        if self.batch_size == "full":
            self.batch_size_inner = n_inner_samples
            self.batch_size_outer = n_outer_samples
        else:
            self.batch_size_inner = self.batch_size
            self.batch_size_outer = self.batch_size

        if self.framework == 'numba':
            # JIT necessary functions and classes
            njit_f2sa = njit(_f2sa)
            self.inner_loop = njit(inner_f2sa)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

            def f2sa(*args, **kwargs):
                return njit_f2sa(self.inner_loop, *args, **kwargs)
            self.f2sa = f2sa
        elif self.framework == "none":
            self.inner_loop = inner_f2sa
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

            def f2sa(*args, **kwargs):
                return _f2sa(self.inner_loop, *args, **kwargs)
            self.f2sa = f2sa
        elif self.framework == 'jax':
            self.f_inner = jax.jit(
                partial(self.f_inner, batch_size=self.batch_size_inner)
            )
            self.f_outer = jax.jit(
                partial(self.f_outer, batch_size=self.batch_size_outer)
            )
            inner_sampler, self.state_inner_sampler \
                = init_sampler(n_samples=n_inner_samples,
                               batch_size=self.batch_size_inner)
            outer_sampler, self.state_outer_sampler \
                = init_sampler(n_samples=n_outer_samples,
                               batch_size=self.batch_size_outer)
            self.f2sa = partial(
                f2sa_jax,
                inner_sampler=inner_sampler,
                outer_sampler=outer_sampler
            )
        else:
            raise ValueError(f"Framework {self.framework} not supported.")

        self.inner_var = inner_var0
        self.outer_var = outer_var0
        self.memory = 0

    def warm_up(self):
        if self.framework in ['numba', 'jax']:
            self.run_once(2)

    def run(self, callback):
        eval_freq = self.eval_freq

        # Init variables
        memory_start = get_memory()
        inner_var = self.inner_var.copy()
        lagrangian_inner_var = self.inner_var.copy()
        outer_var = self.outer_var.copy()
        lmbda = self.lmbda0
        if self.framework == "jax":
            v = jnp.zeros_like(inner_var)
            # Init lr scheduler
            step_sizes = jnp.array(
                [self.step_size, self.step_size / self.outer_ratio]
            )
            exponents = jnp.array(
                [.5, .5]
            )
            state_lr = init_lr_scheduler(step_sizes, exponents)
            carry = dict(
                state_lr=state_lr,
                state_inner_sampler=self.state_inner_sampler,
                state_outer_sampler=self.state_outer_sampler,
            )
        else:
            rng = np.random.RandomState(self.random_state)
            v = np.zeros_like(inner_var)
            # Init lr scheduler
            step_sizes = np.array(
                [self.step_size,
                 self.step_size,
                 self.step_size / self.outer_ratio,
                 self.delta_lmbda]
            )
            exponents = np.array(
                [.5, .5, .5, 0]
            )
            lr_scheduler = self.LearningRateScheduler(
                np.array(step_sizes, dtype=float), exponents
            )
            inner_sampler = self.MinibatchSampler(self.n_inner_samples,
                                                  self.batch_size_inner)
            outer_sampler = self.MinibatchSampler(self.n_outer_samples,
                                                  self.batch_size_outer)

        # Start algorithm
        while callback():
            if self.framework == 'jax':
                inner_var, outer_var, v, carry = self.f2sa(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v, max_iter=eval_freq, **carry
                )
            else:
                inner_var, outer_var, lagrangian_inner_var, lmbda = self.f2sa(
                    self.f_inner, self.f_outer, inner_var, outer_var,
                    lagrangian_inner_var, lmbda, inner_sampler=inner_sampler,
                    outer_sampler=outer_sampler, lr_scheduler=lr_scheduler,
                    n_inner_steps=self.n_inner_steps, max_iter=eval_freq,
                    seed=rng.randint(constants.MAX_SEED)
                )
            memory_end = get_memory()
            self.inner_var = inner_var
            self.outer_var = outer_var
            self.memory = memory_end - memory_start
            self.memory /= 1e6

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var,
                    memory=self.memory)


def inner_f2sa(inner_oracle, outer_oracle, inner_var, lagrangian_inner_var,
               outer_var, lmbda, inner_sampler=None, outer_sampler=None,
               lr_inner=.1, lr_lagrangian=.1, n_steps=10):
    """
    Inner loop of F2SA algorithm.
    """
    for _ in range(n_steps):
        # Get the batches and oracles
        slice_inner, _ = inner_sampler.get_batch()
        slice_inner_lagrangian, _ = inner_sampler.get_batch()
        slice_outer, _ = outer_sampler.get_batch()

        d_inner_var = inner_oracle.grad_inner_var(inner_var, outer_var,
                                                  slice_inner)
        d_lagrangian_inner_var = lmbda * inner_oracle.grad_inner_var(
            lagrangian_inner_var, outer_var, slice_inner_lagrangian
        )
        d_lagrangian_inner_var += outer_oracle.grad_inner_var(
            inner_var, outer_var, slice_outer
        )

        # Update the variables
        inner_var -= lr_inner * d_inner_var
        lagrangian_inner_var -= lr_lagrangian * d_lagrangian_inner_var
    return inner_var, lagrangian_inner_var


def _f2sa(inner_loop, inner_oracle, outer_oracle, inner_var, outer_var,
          lagrangian_inner_var, lmbda, inner_sampler=None, outer_sampler=None,
          lr_scheduler=None, n_inner_steps=10, max_iter=1, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        lr_inner, lr_lagrangian, lr_outer, d_lmbda = lr_scheduler.get_lr()

        # Run the inner procedure
        inner_var, lagrangian_inner_var = inner_loop(
            inner_oracle, outer_oracle, inner_var, lagrangian_inner_var,
            outer_var, lmbda, inner_sampler=inner_sampler,
            outer_sampler=outer_sampler, lr_inner=lr_inner,
            lr_lagrangian=lr_lagrangian, n_steps=n_inner_steps
        )

        # Compute oracles
        slice_outer, _ = outer_sampler.get_batch()
        slice_inner1, _ = inner_sampler.get_batch()
        slice_inner2, _ = inner_sampler.get_batch()

        d_outer_var = outer_oracle.grad_outer_var(inner_var, outer_var,
                                                  slice_outer)
        grad_inner = inner_oracle.grad_outer_var(inner_var, outer_var,
                                                 slice_inner1)
        grad_inner_star = inner_oracle.grad_outer_var(lagrangian_inner_var,
                                                      outer_var, slice_inner2)

        d_outer_var += lmbda * (grad_inner_star - grad_inner)

        # Step.2 - update the variables
        outer_var -= lr_outer * d_outer_var
        lmbda += d_lmbda

    return inner_var, outer_var, lagrangian_inner_var, lmbda


@partial(jax.jit, static_argnames=('inner_sampler', 'outer_sampler', 'n_steps',
                                   'grad_inner', 'grad_outer'))
def inner_f2sa_jax(inner_var, lagrangian_inner_var,  outer_var, lmbda,
                   state_inner_sampler, state_outer_sampler,
                   inner_sampler=None, outer_sampler=None,
                   lr_inner=.1, lr_lagrangian=.1, n_steps=10, grad_inner=None,
                   grad_outer=None):
    """
    Jax implementation of the inner loop of F2SA algorithm.
    """
    def iter(i, args):
        (inner_var, lagrangian_inner_var, state_inner_sampler,
         state_outer_sampler) = args
        # Get the batches and oracles
        slice_inner, _ = inner_sampler.get_batch()
        start_idx_inner, *_, state_inner_sampler = inner_sampler(
            state_inner_sampler
        )
        start_idx_lagrangian, *_, state_inner_sampler = inner_sampler(
            state_inner_sampler
        )
        start_idx_outer, *_, state_outer_sampler = outer_sampler(
            state_outer_sampler
        )

        d_inner_var = grad_inner(inner_var, outer_var, start_idx_inner)
        d_lagrangian_inner_var = lmbda * grad_inner(
            lagrangian_inner_var, outer_var, start_idx_lagrangian
        )
        d_lagrangian_inner_var += grad_outer(
            inner_var, outer_var, start_idx_outer
        )

        # Update the variables
        inner_var -= lr_inner * d_inner_var
        lagrangian_inner_var -= lr_lagrangian * d_lagrangian_inner_var
        return (inner_var, lagrangian_inner_var, state_inner_sampler,
                state_outer_sampler)
    (inner_var, lagrangian_inner_var, state_inner_sampler,
     state_outer_sampler) = jax.lax.fori_loop(
        0, n_steps, iter, (inner_var, lagrangian_inner_var,
                           state_inner_sampler, state_outer_sampler)
    )
    return (inner_var, lagrangian_inner_var, state_inner_sampler,
            state_outer_sampler)


@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('inner_sampler', 'outer_sampler', 'max_iter'))
def f2sa_jax(f_inner, f_outer, inner_var, outer_var, v,
             state_inner_sampler=None, state_outer_sampler=None, state_lr=None,
             inner_sampler=None, outer_sampler=None, max_iter=1):

    grad_inner = jax.grad(f_inner, argnums=0)
    grad_outer_inner_var = jax.grad(f_outer, argnums=0)
    grad_outer_outer_var = jax.grad(f_outer, argnums=1)

    def f2sa_one_iter(carry, _):

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

        # #Use prox to make sure we do not diverge
        # # inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        return carry, _

    init = dict(
        inner_var=inner_var, outer_var=outer_var, v=v, state_lr=state_lr,
        state_inner_sampler=state_inner_sampler,
        state_outer_sampler=state_outer_sampler
    )
    carry, _ = jax.lax.scan(
        f2sa_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )

    return (
        carry['inner_var'], carry['outer_var'], carry['v'],
        {k: v for k, v in carry.items()
         if k not in ['inner_var', 'outer_var', 'v']}
    )
