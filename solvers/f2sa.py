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
        'step_size': [.01],
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
        elif self.framework not in ['none', 'numba', 'jax']:
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
            self.inner_loop = partial(
                inner_f2sa_jax,
                inner_sampler=inner_sampler,
                outer_sampler=outer_sampler,
                grad_inner=jax.grad(self.f_inner, argnums=0),
                grad_outer=jax.grad(self.f_outer, argnums=0)
            )
            self.f2sa = partial(
                f2sa_jax,
                inner_f2sa=self.inner_loop,
                inner_sampler=inner_sampler,
                outer_sampler=outer_sampler
            )
        else:
            raise ValueError(f"Framework {self.framework} not supported.")

        self.inner_var = inner_var0
        self.outer_var = outer_var0
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.memory = 0

    def warm_up(self):
        if self.framework in ['numba', 'jax']:
            self.run_once(2)
            self.inner_var = self.inner_var0
            self.outer_var = self.outer_var0

    def run(self, callback):
        eval_freq = self.eval_freq

        # Init variables
        memory_start = get_memory()
        inner_var = self.inner_var.copy()
        inner_approx_star = self.inner_var.copy()
        outer_var = self.outer_var.copy()
        lmbda = self.lmbda0
        if self.framework == "jax":
            # Init lr scheduler
            step_sizes = jnp.array(
                [self.step_size,
                 self.step_size,
                 self.step_size / self.outer_ratio,
                 self.delta_lmbda]
            )
            exponents = jnp.array(
                [5/7, 4/7, 4/7, 1/7]
            )
            state_lr = init_lr_scheduler(step_sizes, exponents)
            carry = dict(
                state_lr=state_lr,
                state_inner_sampler=self.state_inner_sampler,
                state_outer_sampler=self.state_outer_sampler,
            )
        else:
            rng = np.random.RandomState(self.random_state)
            # Init lr scheduler
            step_sizes = np.array(
                [self.step_size,
                 self.step_size,
                 self.step_size / self.outer_ratio,
                 self.delta_lmbda]
            )
            exponents = np.array(
                [5/7, 4/7, 4/7, 1/7]
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
                inner_var, outer_var, inner_approx_star, lmbda, \
                    carry = self.f2sa(
                        self.f_inner, self.f_outer,
                        inner_var, outer_var, inner_approx_star, lmbda,
                        n_inner_steps=self.n_inner_steps,
                        max_iter=eval_freq, **carry
                    )
            else:
                inner_var, outer_var, inner_approx_star, lmbda = self.f2sa(
                    self.f_inner, self.f_outer, inner_var, outer_var,
                    inner_approx_star, lmbda, inner_sampler=inner_sampler,
                    outer_sampler=outer_sampler, lr_scheduler=lr_scheduler,
                    n_inner_steps=self.n_inner_steps, max_iter=eval_freq,
                    seed=rng.randint(constants.MAX_SEED)
                )
            memory_end = get_memory()
            self.inner_var = inner_approx_star
            self.outer_var = outer_var
            self.memory = memory_end - memory_start
            self.memory /= 1e6

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var,
                    memory=self.memory)


def inner_f2sa(inner_oracle, outer_oracle, inner_var, inner_approx_star,
               outer_var, lmbda, inner_sampler=None, outer_sampler=None,
               lr_inner=.1, lr_approx_star=.1, n_steps=10):
    """
    Inner loop of F2SA algorithm.

    Parameters
    ----------
    inner_oracle, outer_oracle : oracle classes
        Inner and outer oracles.

    inner_var : array, shape (d_inner,)
        Initial inner variable.

    inner_approx_star : array, shape (d_inner,)
        Initial inner variable.

    outer_var : array, shape (d_outer,)
        Outer variable.

    lmbda : float
        Lagrange multiplier.

    inner_sampler, outer_sampler : MiniBatchSampler
        Inner sampler.

    lr_inner : float
        Learning rate for the inner variable.

    lr_approx_star : float
        Learning rate for the lagrangian inner variable.

    n_steps : int
        Number of steps of the loop.

    Returns
    -------
    inner_var : array, shape (d_inner,)
        Updated inner variable.

    inner_approx_star : array, shape (d_inner,)
        Updated inner variable to approximate g^*.
    """
    for _ in range(n_steps):
        # Get the batches and oracles
        slice_inner, _ = inner_sampler.get_batch()
        slice_inner_lagrangian, _ = inner_sampler.get_batch()
        slice_outer, _ = outer_sampler.get_batch()

        d_inner_var = lmbda * inner_oracle.grad_inner_var(inner_var, outer_var,
                                                          slice_inner)
        d_inner_var += outer_oracle.grad_inner_var(
            inner_var, outer_var, slice_outer
        )
        d_inner_approx_star = inner_oracle.grad_inner_var(
            inner_approx_star, outer_var, slice_inner_lagrangian
        )

        # Update the variables
        inner_var -= lr_inner * d_inner_var
        inner_approx_star -= lr_approx_star * d_inner_approx_star
    return inner_var, inner_approx_star


def _f2sa(inner_loop, inner_oracle, outer_oracle, inner_var, outer_var,
          inner_approx_star, lmbda, inner_sampler=None, outer_sampler=None,
          lr_scheduler=None, n_inner_steps=10, max_iter=1, seed=None):
    """
    Implementation of the F2SA algorithm.

    Parameters
    ----------
    inner_loop : callable
        Inner loop of F2SA algorithm.

    inner_oracle, outer_oracle : oracle classes
        Inner and outer oracles.

    inner_var : array, shape (d_inner,)
        Initial inner variable.

    outer_var : array, shape (d_outer,)
        Outer variable.

    inner_approx_star : array, shape (d_inner,)
        Initial inner variable.

    lmbda : float
        Lagrange multiplier.

    n_inner_steps : int
        Number of steps of the inner loop.

    lr_scheduler : LearningRateScheduler
        Learning rate scheduler.

    inner_sampler, outer_sampler : MiniBatchSampler
        Inner and outer samplers.

    max_iter : int
        Number of iterations of the outer loop.

    seed : int
        Seed for randomness.

    Returns
    -------
    inner_var : array, shape (d_inner,)
        Updated inner variable.

    outer_var : array, shape (d_outer,)
        Updated outer variable.

    inner_approx_star : array, shape (d_inner,)
        Updated inner variable  to approximate g^*.

    lmbda : float
        Updated Lagrange multiplier.

    state_inner_sampler : dict
        Updated state of the inner sampler.

    state_outer_sampler : dict
        Updated state of the outer sampler.

    state_lr : dict
        Updated state of the learning rate scheduler.

    """

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        lr_inner, lr_approx_star, lr_outer, d_lmbda = lr_scheduler.get_lr()

        # Run the inner procedure
        inner_var, inner_approx_star = inner_loop(
            inner_oracle, outer_oracle, inner_var, inner_approx_star,
            outer_var, lmbda, inner_sampler=inner_sampler,
            outer_sampler=outer_sampler, lr_inner=lr_inner,
            lr_approx_star=lr_approx_star, n_steps=n_inner_steps
        )

        # Compute oracles
        slice_outer, _ = outer_sampler.get_batch()
        slice_inner1, _ = inner_sampler.get_batch()
        slice_inner2, _ = inner_sampler.get_batch()

        d_outer_var = outer_oracle.grad_outer_var(inner_var, outer_var,
                                                  slice_outer)
        grad_inner = inner_oracle.grad_outer_var(inner_var, outer_var,
                                                 slice_inner1)
        grad_inner_star = inner_oracle.grad_outer_var(inner_approx_star,
                                                      outer_var, slice_inner2)

        d_outer_var += lmbda * (grad_inner - grad_inner_star)

        # Step.2 - update the variables
        outer_var -= lr_outer * d_outer_var
        lmbda += d_lmbda

    return inner_var, outer_var, inner_approx_star, lmbda


@partial(jax.jit, static_argnames=('inner_sampler', 'outer_sampler', 'n_steps',
                                   'grad_inner', 'grad_outer'))
def inner_f2sa_jax(inner_var, inner_approx_star,  outer_var, lmbda,
                   state_inner_sampler, state_outer_sampler,
                   inner_sampler=None, outer_sampler=None,
                   lr_inner=.1, lr_approx_star=.1, n_steps=10, grad_inner=None,
                   grad_outer=None):
    """
    Jax implementation of the inner loop of F2SA algorithm.

    Parameters
    ----------
    inner_var : array, shape (d_inner,)
        Initial inner variable.

    inner_approx_star : array, shape (d_inner,)
        Initial inner variable.

    outer_var : array, shape (d_outer,)
        Outer variable.

    lmbda : float
        Lagrange multiplier.

    state_inner_sampler : dict
        State of the inner sampler.

    state_outer_sampler : dict
        State of the outer sampler.

    inner_sampler : callable
        Inner sampler.

    outer_sampler : callable
        Outer sampler.

    lr_inner : float
        Learning rate for the inner variable.

    lr_approx_star : float
        Learning rate for the lagrangian inner variable.

    n_steps : int
        Number of steps of the loop.

    grad_inner : callable
        Gradient of the inner oracle with respect to the inner variable.

    grad_outer : callable
        Gradient of the outer oracle with respect to the inner variable.

    Returns
    -------
    inner_var : array, shape (d_inner,)
        Updated inner variable.

    inner_approx_star : array, shape (d_inner,)
        Updated inner variable to approximate g^*.

    state_inner_sampler : dict
        Updated state of the inner sampler.

    state_outer_sampler : dict
        Updated state of the outer sampler.
    """
    def iter(i, args):
        (inner_var, inner_approx_star, state_inner_sampler,
         state_outer_sampler) = args
        # Get the batches and oracles
        start_idx_inner, *_, state_inner_sampler = inner_sampler(
            state_inner_sampler
        )
        start_idx_lagrangian, *_, state_inner_sampler = inner_sampler(
            state_inner_sampler
        )
        start_idx_outer, *_, state_outer_sampler = outer_sampler(
            state_outer_sampler
        )

        d_inner_var = lmbda * grad_inner(
            inner_var, outer_var, start_idx_inner
        )
        d_inner_var += grad_outer(inner_var, outer_var, start_idx_outer)
        d_inner_approx_star = grad_inner(
            inner_approx_star, outer_var, start_idx_lagrangian
        )

        # # Update the variables
        inner_var -= lr_inner * d_inner_var
        inner_approx_star -= lr_approx_star * d_inner_approx_star
        return (inner_var, inner_approx_star, state_inner_sampler,
                state_outer_sampler)
    (inner_var, inner_approx_star, state_inner_sampler,
     state_outer_sampler) = jax.lax.fori_loop(
        0, n_steps, iter, (inner_var, inner_approx_star,
                           state_inner_sampler, state_outer_sampler)
    )
    return (inner_var, inner_approx_star, state_inner_sampler,
            state_outer_sampler)


@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('inner_sampler', 'outer_sampler', 'max_iter',
                          'n_inner_steps', 'inner_f2sa'))
def f2sa_jax(f_inner, f_outer, inner_var, outer_var, inner_approx_star,
             lmbda, state_inner_sampler=None, state_outer_sampler=None,
             state_lr=None, inner_f2sa=None, n_inner_steps=1,
             inner_sampler=None, outer_sampler=None, max_iter=1):
    """
    Jax implementation of the F2SA algorithm.

    Parameters
    ----------
    f_inner, f_outer : callables
        Inner and outer oracles.

    inner_var : array, shape (d_inner,)
        Initial inner variable.

    outer_var : array, shape (d_outer,)
        Outer variable.

    inner_approx_star : array, shape (d_inner,)
        Initial inner variable to approximate g^*.

    lmbda : float
        Lagrange multiplier.

    state_inner_sampler : dict
        State of the inner sampler.

    state_outer_sampler : dict
        State of the outer sampler.

    state_lr : dict
        State of the learning rate scheduler.

    inner_f2sa : callable
        Inner loop of F2SA algorithm.

    n_inner_steps : int
        Number of steps of the inner loop.

    inner_sampler : callable
        Inner sampler.

    outer_sampler : callable
        Outer sampler.

    max_iter : int
        Number of iterations of the outer loop.

    Returns
    -------
    inner_var : array, shape (d_inner,)
        Updated inner variable.

    outer_var : array, shape (d_outer,)
        Updated outer variable.

    inner_approx_star : array, shape (d_inner,)
        Updated inner variable to approximate g^*.

    lmbda : float
        Updated Lagrange multiplier.

    state_inner_sampler : dict
        Updated state of the inner sampler.

    state_outer_sampler : dict
        Updated state of the outer sampler.

    state_lr : dict
        Updated state of the learning rate scheduler.

    """

    grad_inner = jax.grad(f_inner, argnums=1)
    grad_outer_outer_var = jax.grad(f_outer, argnums=1)

    def f2sa_one_iter(carry, _):

        step_sizes, carry['state_lr'] = update_lr(
            carry['state_lr']
        )
        lr_inner, lr_approx_star, lr_outer, d_lmbda = step_sizes

        # Run the inner procedure
        carry['inner_var'], carry['inner_approx_star'], \
            carry['state_inner_sampler'], carry['state_outer_sampler'] = \
            inner_f2sa(
                carry['inner_var'], carry['inner_approx_star'],
                carry['outer_var'], carry['lmbda'],
                carry['state_inner_sampler'], carry['state_outer_sampler'],
                inner_sampler=inner_sampler, outer_sampler=outer_sampler,
                lr_inner=lr_inner, lr_approx_star=lr_approx_star,
                n_steps=n_inner_steps
            )

        # Compute oracles and get the update direction of the outer variable
        start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
            carry['state_outer_sampler']
        )
        start_inner1, *_, carry['state_inner_sampler'] = inner_sampler(
            carry['state_inner_sampler']
        )
        start_inner2, *_, carry['state_inner_sampler'] = inner_sampler(
            carry['state_inner_sampler']
        )
        d_outer_var = grad_outer_outer_var(
            carry['inner_var'], carry['outer_var'], start_outer
        )
        grad_inner_outer = grad_inner(
            carry['inner_var'], carry['outer_var'], start_inner1
        )
        grad_inner_star = grad_inner(
            carry['inner_approx_star'], carry['outer_var'], start_inner2
        )
        d_outer_var += carry['lmbda'] * (grad_inner_outer - grad_inner_star)

        # Update inner variable with SGD.
        carry['outer_var'] -= lr_outer * d_outer_var
        carry['lmbda'] += d_lmbda

        return carry, _

    init = dict(
        inner_var=inner_var,
        inner_approx_star=inner_approx_star,
        outer_var=outer_var,
        lmbda=lmbda,
        state_lr=state_lr,
        state_inner_sampler=state_inner_sampler,
        state_outer_sampler=state_outer_sampler,
    )
    carry, _ = jax.lax.scan(
        f2sa_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )

    return (
        carry['inner_var'], carry['outer_var'], carry['inner_approx_star'],
        carry['lmbda'],
        {k: v for k, v in carry.items()
         if k not in ['inner_var', 'outer_var', 'inner_approx_star',
                      'lmbda']}
    )
