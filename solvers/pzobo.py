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
    """Partial Zeroth-Order-like Bilevel Optimizer (PZOBO).

    D. Sow, K. Ji and Y. Liang. "On the Convergence Theory for Hessian-Free
    Bilevel Algorithms". arxiv:2110.07004 2022"""
    name = 'PZOBO'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'eval_freq': [1],
        'random_state': [1],
        'mu': [.1],
        'n_inner_steps': [10],
        'n_gaussian_vectors': [1],
        'framework': ["jax"],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def skip(self, f_train, f_val, **kwargs):
        if self.framework == 'numba':
            if isinstance(f_train(), (MultiLogRegOracle, DataCleaningOracle)):
                return True, "Numba implementation not available for " \
                      "this oracle."
            elif isinstance(f_val(), (MultiLogRegOracle, DataCleaningOracle)):
                return True, "Numba implementation not available for" \
                      "this oracle."
        elif self.framework not in ['jax', 'none', 'numba']:
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
        self.f_inner = f_train(framework=self.framework, get_full_batch=True)
        self.f_outer = f_val(framework=self.framework, get_full_batch=True)

        if self.framework == 'numba':
            # JIT necessary functions and classes
            njit_pzobo = njit(_pzobo)
            self.gd_inner = njit(gd_inner)

            def pzobo(*args, **kwargs):
                return njit_pzobo(self.gd_inner, *args, **kwargs)
            self.pzobo = pzobo

            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )
        elif self.framework == "none":
            # JIT necessary functions and classes
            self.gd_inner = gd_inner

            def pzobo(*args, **kwargs):
                return _pzobo(self.gd_inner, *args, **kwargs)
            self.pzobo = pzobo
            self.LearningRateScheduler = LearningRateScheduler
        elif self.framework == 'jax':
            _, self.f_inner = self.f_inner
            _, self.f_outer = self.f_outer
            self.f_inner = jax.jit(self.f_inner)
            self.f_outer = jax.jit(self.f_outer)
            self.pzobo = partial(
                pzobo_jax,
                gd_inner=gd_inner_jax
            )
        else:
            raise ValueError(f"Framework {self.framework} not supported.")

        self.inner_var = inner_var0
        self.outer_var = outer_var0
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

    def warm_up(self):
        if self.framework in ['numba', 'jax']:
            self.run_once(2)
            self.inner_var = self.inner_var0
            self.outer_var = self.outer_var0

    def run(self, callback):
        eval_freq = self.eval_freq
        memory_start = get_memory()

        # Init variables
        inner_var = self.inner_var.copy()
        outer_var = self.outer_var.copy()
        if self.framework == "jax":
            # Init lr scheduler
            step_sizes = jnp.array(
                [self.step_size, self.step_size / self.outer_ratio]
            )
            exponents = jnp.array(
                [0., 0.]
            )
            state_lr = init_lr_scheduler(step_sizes, exponents)
            carry = dict(
                state_lr=state_lr,
                key=jax.random.PRNGKey(self.random_state)
            )
        else:
            rng = np.random.RandomState(self.random_state)
            # Init lr scheduler
            step_sizes = np.array(
                [self.step_size, self.step_size / self.outer_ratio]
            )
            exponents = np.array(
                [0., 0.]
            )
            lr_scheduler = self.LearningRateScheduler(
                np.array(step_sizes, dtype=float), exponents
            )

        # Start algorithm
        while callback():
            if self.framework == 'jax':
                inner_var, outer_var, carry = self.pzobo(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, mu=self.mu,
                    n_inner_steps=self.n_inner_steps,
                    n_gaussian_vectors=self.n_gaussian_vectors,
                    max_iter=eval_freq, **carry
                )
            else:
                inner_var, outer_var = self.pzobo(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, mu=self.mu,
                    n_inner_steps=self.n_inner_steps,
                    n_gaussian_vectors=self.n_gaussian_vectors,
                    lr_scheduler=lr_scheduler, max_iter=eval_freq,
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


def _pzobo(gd_inner, inner_oracle, outer_oracle, inner_var, outer_var, mu=.1,
           n_inner_steps=1, n_gaussian_vectors=1, lr_scheduler=None,
           max_iter=1, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)
    inner_var_shape = inner_var.shape[0]
    outer_var_shape = outer_var.shape[0]
    for i in range(max_iter):
        lr_inner, lr_outer = lr_scheduler.get_lr()
        inner_var_old = inner_var.copy()

        # Update inner variable by GD
        inner_var = gd_inner(inner_oracle, inner_var, outer_var, lr_inner,
                             n_steps=n_inner_steps)

        # Sample gaussian vectors and perform ES estimation
        U = np.random.randn(n_gaussian_vectors, outer_var_shape)
        outer_var_aux = outer_var + mu * U
        deltas = np.zeros((n_gaussian_vectors, inner_var_shape))
        for q in range(n_gaussian_vectors):
            deltas[q] = gd_inner(inner_oracle, inner_var_old,
                                 outer_var_aux[q], lr_inner,
                                 n_steps=n_inner_steps)
            deltas[q] -= inner_var
        deltas = deltas / mu

        grad_outer_in, grad_outer_out = outer_oracle.grad(
            inner_var, outer_var, slice(None)
        )

        es_estimator = U.T.dot(deltas @ grad_outer_in) / n_gaussian_vectors
        outer_var -= lr_outer * (es_estimator + grad_outer_out)

    return inner_var, outer_var


@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('max_iter', 'n_inner_steps', 'n_gaussian_vectors',
                          'gd_inner'))
def pzobo_jax(f_inner, f_outer, inner_var, outer_var, mu=.1,
              state_lr=None, n_inner_steps=1, n_gaussian_vectors=1,
              max_iter=1, key=None, gd_inner=None):

    grad_inner = jax.grad(f_inner, argnums=0)
    grad_outer = jax.grad(f_outer, argnums=(0, 1))

    def pzobo_one_iter(carry, _):
        inner_var_shape = inner_var.shape[0]
        outer_var_shape = outer_var.shape[0]
        carry['key'] = jax.random.split(carry['key'], 1)[0]
        (inner_step_size, outer_step_size), carry['state_lr'] = update_lr(
            carry['state_lr']
        )

        inner_var_old = carry['inner_var'].copy()

        # Update inner variable by GD
        carry['inner_var'] = gd_inner(grad_inner, carry['inner_var'],
                                      carry['outer_var'], inner_step_size,
                                      n_steps=n_inner_steps)

        U = jax.random.normal(carry['key'], (n_gaussian_vectors,
                                             outer_var_shape))
        outer_var_aux = carry['outer_var'] + mu * U

        def iter_fun(q, deltas):
            deltasq = gd_inner(grad_inner, inner_var_old, outer_var_aux[q],
                               inner_step_size, n_steps=n_inner_steps)
            deltas = deltas.at[q].set(deltasq - carry['inner_var'])
            return deltas

        deltas = jax.lax.fori_loop(0, n_gaussian_vectors, iter_fun,
                                   jnp.zeros((n_gaussian_vectors,
                                              inner_var_shape)))
        deltas /= mu
        grad_outer_in, grad_outer_out = grad_outer(carry['inner_var'],
                                                   carry['outer_var'])
        es_estimator = U.T.dot(deltas @ grad_outer_in) / n_gaussian_vectors
        carry['outer_var'] -= outer_step_size * (es_estimator + grad_outer_out)

        return carry, _

    init = dict(
        inner_var=inner_var, outer_var=outer_var, state_lr=state_lr, key=key
    )
    carry, _ = jax.lax.scan(
        pzobo_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )
    return (
        carry['inner_var'], carry['outer_var'],
        {k: v for k, v in carry.items() if k not in ['inner_var', 'outer_var']}
    )
