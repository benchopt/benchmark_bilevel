from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.get_memory import get_memory
    from benchmark_utils.gd_inner import gd_inner, gd_inner_jax
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec
    from benchmark_utils.oracles import MultiLogRegOracle, DataCleaningOracle
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler

    import jax
    import jax.numpy as jnp
    from functools import partial


class Solver(BaseSolver):
    """Bilevel Optimization Made Easy (BOME).

    M. Ye, B. Liu, S. Wright, P. Stone and Q. Liu, "BOME! Bilevel Optimization
    Made Easy: A Simple First-Order Approach", NeurIPS 2022."""
    name = 'BOME'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'eval_freq': [1],
        'random_state': [1],
        'framework': ["jax"],
        'choice_phi': ["grad_norm"],
        'eta': [5e-1],
        'n_inner_steps': [10],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def skip(self, f_train, f_val, **kwargs):
        if self.framework == 'numba':
            if isinstance(f_train(),
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

        if self.framework == 'numba':
            # JIT necessary functions and classes
            njit_bome = njit(_bome)
            self.inner_loop = njit(gd_inner)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

            def bome(*args, **kwargs):
                return njit_bome(self.inner_loop, *args, **kwargs)
            self.bome = bome
        elif self.framework == "none":
            self.inner_loop = gd_inner
            self.LearningRateScheduler = LearningRateScheduler

            def bome(*args, **kwargs):
                return _bome(self.inner_loop, *args, **kwargs)
            self.bome = bome
        elif self.framework == 'jax':
            self.n_inner_samples = n_inner_samples
            self.n_outer_samples = n_outer_samples

            self.f_inner = jax.jit(
                partial(f_train(framework='jax'),
                        batch_size=n_inner_samples, start=0),
            )
            self.f_outer = jax.jit(
                partial(f_val(framework='jax'),
                        batch_size=n_outer_samples, start=0),
            )
            self.inner_loop = partial(
                gd_inner_jax,
                grad_inner=jax.grad(self.f_inner, argnums=0)
            )
            self.bome = partial(
                bome_jax,
                inner_bome=self.inner_loop
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
        outer_var = self.outer_var.copy()
        if self.framework == "jax":
            # Init lr scheduler
            step_sizes = jnp.array(
                [self.step_size, self.step_size / self.outer_ratio]
            )
            exponents = jnp.array([0, 0])
            state_lr = init_lr_scheduler(step_sizes, exponents)
            choice_phi_dict = dict(
                grad_norm=0,
                inner_suboptimality=1
            )
        else:
            # Init lr scheduler
            step_sizes = np.array(
                [self.step_size, self.step_size / self.outer_ratio]
            )
            exponents = np.array(
                [0, 0]
            )
            lr_scheduler = self.LearningRateScheduler(
                np.array(step_sizes, dtype=float), exponents
            )

        # Start algorithm
        while callback():
            if self.framework == 'jax':
                inner_var, outer_var, state_lr = self.bome(
                        self.f_inner, self.f_outer, inner_var, outer_var,
                        choice_phi=choice_phi_dict[self.choice_phi],
                        eta=self.eta, n_inner_steps=self.n_inner_steps,
                        max_iter=eval_freq, state_lr=state_lr
                    )
            else:
                inner_var, outer_var = self.bome(
                    self.f_inner, self.f_outer, inner_var, outer_var,
                    choice_phi=self.choice_phi, eta=self.eta,
                    lr_scheduler=lr_scheduler,
                    n_inner_steps=self.n_inner_steps, max_iter=eval_freq
                )
            memory_end = get_memory()
            self.inner_var = inner_var
            self.outer_var = outer_var
            self.memory = memory_end - memory_start
            self.memory /= 1e6

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var,
                    memory=self.memory)


def _bome(inner_loop, inner_oracle, outer_oracle, inner_var, outer_var,
          choice_phi="grad_norm", eta=5e-1, lr_scheduler=None,
          n_inner_steps=10, max_iter=1):
    """
    Implementation of the BOME algorithm.

    Parameters
    ----------
    inner_loop : callable
        Inner loop of BOME algorithm.

    inner_oracle, outer_oracle : oracle classes
        Inner and outer oracles.

    inner_var : array, shape (d_inner,)
        Initial inner variable.

    outer_var : array, shape (d_outer,)
        Outer variable.

    choice_phi : either "grad_norm" or "inner_suboptimality"
        if "grad_norm", phi is the squared norm of the gradient of the
        subptimality q multiplied by eta.
        if "inner_suboptimality", phi is the suboptimality of the inner oracle
        multiplied by eta.

    eta : float
        Multiplicative factor of phi.

    lr_scheduler : LearningRateScheduler
        Learning rate scheduler.

    n_inner_steps : int
        Number of steps of the inner loop.

    max_iter : int
        Number of iterations of the outer loop.

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

    for i in range(max_iter):
        lr_inner, lr_outer = lr_scheduler.get_lr()

        # Run the inner procedure
        inner_var_star = inner_loop(
            inner_oracle=inner_oracle,
            inner_var=inner_var.copy(),
            outer_var=outer_var,
            step_size=lr_inner,
            n_steps=n_inner_steps
        )

        # Compute oracles
        grad_outer_inner_var, grad_outer_outer_var = outer_oracle.grad(
            inner_var, outer_var, slice(None)
        )
        grad_q_inner_var, grad_q_outer_var = inner_oracle.grad(
            inner_var, outer_var, slice(None)
        )
        grad_q_outer_var -= inner_oracle.grad_outer_var(inner_var_star,
                                                        outer_var,
                                                        slice(None))

        # Compute phi and lmbda
        squared_norm_grad_q = np.linalg.norm(grad_q_inner_var)**2
        squared_norm_grad_q += np.linalg.norm(grad_q_outer_var)**2
        if choice_phi == "grad_norm":
            phi = squared_norm_grad_q
        elif choice_phi == "inner_suboptimality":
            phi = inner_oracle.value(inner_var, outer_var, slice(None))
            phi -= inner_oracle.value(inner_var_star, outer_var, slice(None))
        phi *= eta
        dot_grad = np.dot(grad_outer_inner_var, grad_q_inner_var)
        dot_grad += np.dot(grad_outer_outer_var, grad_q_outer_var)
        lmbda = np.maximum(phi - dot_grad, 0) / squared_norm_grad_q

        # Compute the update direction of the inner and outer variables
        d_inner = grad_outer_inner_var + lmbda * grad_q_inner_var
        d_outer = grad_outer_outer_var + lmbda * grad_q_outer_var

        # Update inner and outer variables
        inner_var -= lr_inner * d_inner
        outer_var -= lr_outer * d_outer
    return inner_var, outer_var


@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('choice_phi', 'max_iter', 'n_inner_steps',
                          'inner_bome'))
def bome_jax(f_inner, f_outer, inner_var, outer_var, choice_phi=0,
             eta=5e-1, state_lr=None, inner_bome=None, n_inner_steps=1,
             max_iter=1):
    """
    Jax implementation of the BOME algorithm.

    Parameters
    ----------
    f_inner, f_outer : callables
        Inner and outer oracles.

    inner_var : array, shape (d_inner,)
        Initial inner variable.

    outer_var : array, shape (d_outer,)
        Outer variable.

    choice_phi : int
        if 0, phi is the squared norm of the gradient of the
        subptimality q multiplied by eta.
        if 1, phi is the suboptimality of the inner oracle
        multiplied by eta.

    eta : float
        Multiplicative factor of phi.

    state_lr : dict
        State of the learning rate scheduler.

    inner_bome : callable
        Inner loop of BOME algorithm.

    n_inner_steps : int
        Number of steps of the inner loop.

    max_iter : int
        Number of iterations of the outer loop.

    Returns
    -------
    inner_var : array, shape (d_inner,)
        Updated inner variable.

    outer_var : array, shape (d_outer,)
        Updated outer variable.

    state_lr : dict
        Updated state of the learning rate scheduler.

    """

    grad_inner = jax.grad(f_inner, argnums=(0, 1))
    grad_outer = jax.grad(f_outer, argnums=(0, 1))

    def bome_one_iter(carry, _):

        step_sizes, carry['state_lr'] = update_lr(
            carry['state_lr']
        )
        lr_inner, lr_outer = step_sizes

        # Run the inner procedure
        inner_var_star = inner_bome(carry['inner_var'],
                                    carry['outer_var'],
                                    lr_inner,
                                    n_steps=n_inner_steps)

        # Compute oracles
        grad_outer_inner_var, grad_outer_outer_var = grad_outer(
            carry['inner_var'], carry['outer_var']
        )
        grad_q_inner_var, grad_q_outer_var = grad_inner(
            carry['inner_var'], carry['outer_var']
        )
        grad_q_outer_var -= grad_inner(
            inner_var_star, carry['outer_var']
        )[1]

        # Compute phi and lmbda
        squared_norm_grad_q = jnp.linalg.norm(grad_q_inner_var)**2
        squared_norm_grad_q += jnp.linalg.norm(grad_q_outer_var)**2
        phi = jax.lax.cond(
            choice_phi == 0,
            lambda _: squared_norm_grad_q,
            lambda _: f_inner(carry['inner_var'], carry['outer_var'])
            - f_inner(inner_var_star, carry['outer_var']),
            None
        )
        phi *= carry['eta']
        dot_grad = jnp.dot(grad_outer_inner_var, grad_q_inner_var)
        dot_grad += jnp.dot(grad_outer_outer_var, grad_q_outer_var)
        lmbda = jnp.maximum(phi - dot_grad, 0) / squared_norm_grad_q

        # Compute the update direction of the inner and outer variables
        d_inner = grad_outer_inner_var + lmbda * grad_q_inner_var
        d_outer = grad_outer_outer_var + lmbda * grad_q_outer_var

        # Update inner and outer variables
        carry['inner_var'] -= lr_inner * d_inner
        carry['outer_var'] -= lr_outer * d_outer
        return carry, _

    init = dict(
        inner_var=inner_var,
        outer_var=outer_var,
        eta=eta,
        choise_phi=choice_phi,
        state_lr=state_lr,
    )
    carry, _ = jax.lax.scan(
        bome_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )

    return carry['inner_var'], carry['outer_var'], carry['state_lr']
