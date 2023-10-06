
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
    from benchmark_utils.hessian_approximation import hia, hia_jax
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler
    from benchmark_utils.oracles import MultiLogRegOracle, DataCleaningOracle

    import jax
    import jax.numpy as jnp
    from functools import partial


class Solver(BaseSolver):
    """Two-Timescale Stochastic Approximation (TTSA).

    M. Hong, H.-T. Wai and Z. Yang. "A Two-Timescale Framework for Bilevel
    Optimization: Complexity Analysis and Application to Actor-Critic". SIAM
    Journal of Optimization. 2023"""
    name = 'TTSA'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_hia_steps': [10],
        'batch_size': [64],
        'eval_freq': [128],
        'random_state': [1],
        'framework': ["jax"]
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
        self.f_inner = f_train(framework=self.framework)
        self.f_outer = f_val(framework=self.framework)
        self.n_inner_samples = n_inner_samples
        self.n_outer_samples = n_outer_samples

        # Init sampler and lr scheduler
        if self.batch_size == "full":
            self.batch_size_inner = n_inner_samples
            self.batch_size_outer = n_outer_samples
        else:
            self.batch_size_inner = self.batch_size
            self.batch_size_outer = self.batch_size

        if self.framework == 'numba':
            # JIT necessary functions and classes
            njit_hia = njit(hia)
            njit_ttsa = njit(_ttsa)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

            def ttsa(*args, **kwargs):
                return njit_ttsa(njit_hia, *args, **kwargs)
            self.ttsa = ttsa
        elif self.framework == 'none':
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

            def ttsa(*args, **kwargs):
                return _ttsa(hia, *args, **kwargs)
            self.ttsa = ttsa
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
            self.ttsa = partial(
                ttsa_jax,
                hia=hia_jax,
                inner_sampler=inner_sampler,
                outer_sampler=outer_sampler
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
        if self.framework == 'jax':
            step_sizes = jnp.array(
                [self.step_size, self.step_size,
                 self.step_size / self.outer_ratio]
            )
            exponents = jnp.array([.4, 0., .6])
            state_lr = init_lr_scheduler(step_sizes, exponents)
            carry = dict(
                state_lr=state_lr,
                state_inner_sampler=self.state_inner_sampler,
                state_outer_sampler=self.state_outer_sampler,
                key=jax.random.PRNGKey(self.random_state)
            )
        else:
            rng = np.random.RandomState(self.random_state)
            inner_sampler = self.MinibatchSampler(
                self.f_inner.n_samples, batch_size=self.batch_size_inner
            )
            outer_sampler = self.MinibatchSampler(
                self.f_outer.n_samples, batch_size=self.batch_size_outer
            )
            step_sizes = np.array(
                [self.step_size, self.step_size,
                 self.step_size / self.outer_ratio]
            )
            exponents = np.array([.4, 0., .6])
            lr_scheduler = self.LearningRateScheduler(
                np.array(step_sizes, dtype=float), exponents
            )

        while callback():
            if self.framework == 'jax':
                inner_var, outer_var, carry = self.ttsa(
                        self.f_inner, self.f_outer, inner_var, outer_var,
                        n_hia_steps=self.n_hia_steps, max_iter=eval_freq,
                        **carry
                    )
            else:
                inner_var, outer_var, = self.ttsa(
                    self.f_inner, self.f_outer, inner_var, outer_var,
                    lr_scheduler, inner_sampler, outer_sampler,
                    n_hia_steps=self.n_hia_steps, max_iter=eval_freq,
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


def _ttsa(
    hia, inner_oracle, outer_oracle, inner_var, outer_var, lr_scheduler,
    inner_sampler, outer_sampler, n_hia_steps=1, max_iter=1, seed=None
):
    """Numba compatible TTSA algorithm.

    Parameters
    ----------
    inner_oracle, outer_oracle: NumbaOracle
        Inner and outer problem oracles used to compute gradients, etc...
    inner_var, outer_var: ndarray
        Current estimates of the inner and outer variables of the bi-level
        problem.
    max_iter: int
        Maximal number of iterations for the outer problem.
    outer_step_size: float
        Step size to update the outer variable.
    inner_step_size: float
        Step size to update the inner variable.
    n_hia_step: int
        Maximal number of iteration for the HIA problem.
    hia_step_size: float
        Step size for the HIA sub-routine.
    inner_sampler, outer_sampler: MinibatchSampler
        Sampler to get minibatch in a fast and efficient way for the inner and
        outer problems.
    """

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_lr, hia_lr, outer_lr = lr_scheduler.get_lr()

        # Step.1 - Update direction for z with momentum
        slice_inner, _ = inner_sampler.get_batch()
        grad_inner_var = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice_inner
        )

        # Step.2 - Update the inner variable
        inner_var -= inner_lr * grad_inner_var

        # Step.3 - Compute implicit grad approximation with HIA
        slice_outer, _ = outer_sampler.get_batch()
        grad_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        ihvp = hia(
            inner_oracle, inner_var, outer_var, grad_outer,
            hia_lr, sampler=inner_sampler, n_steps=n_hia_steps
        )
        impl_grad -= inner_oracle.cross(
            inner_var, outer_var, ihvp, slice_inner
        )

        # Step.4 - update the outer variables
        outer_var -= outer_lr * impl_grad

        # Step.6 - project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)
    return inner_var, outer_var


@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('hia', 'sgd_inner', 'n_hia_steps', 'n_inner_steps',
                          'inner_sampler', 'outer_sampler', 'max_iter'))
def ttsa_jax(f_inner, f_outer, inner_var, outer_var,
             state_inner_sampler=None, state_outer_sampler=None,
             state_lr=None, hia=None, sgd_inner=None, n_hia_steps=1,
             n_inner_steps=1, inner_sampler=None, outer_sampler=None, key=None,
             max_iter=1):
    grad_inner_fun = jax.grad(f_inner, argnums=0)
    grad_outer_fun = jax.grad(f_outer, argnums=(0, 1))

    def ttsa_one_iter(carry, _):

        (inner_lr, hia_lr, outer_lr), carry['state_lr'] = update_lr(
            carry['state_lr']
        )

        # Step.1 - Update direction for z with momentum
        start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
            carry['state_inner_sampler']
        )
        grad_inner_var = grad_inner_fun(
            carry['inner_var'], carry['outer_var'],
            start_inner
        )

        # Step.2 - Update the inner variable
        carry['inner_var'] -= inner_lr * grad_inner_var

        # Step.3 - Compute implicit grad approximation with HIA
        start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
            carry['state_outer_sampler']
        )
        grad_in, grad_out = grad_outer_fun(
            carry['inner_var'], carry['outer_var'], start_outer
        )

        v, key, carry['state_inner_sampler'] = hia(
            carry['inner_var'], carry['outer_var'], grad_in,
            carry['state_inner_sampler'],
            hia_lr, n_steps=n_hia_steps, sampler=inner_sampler,
            key=carry['key'], grad_inner=grad_inner_fun
        )

        _, vjp_fun = jax.vjp(
            lambda x: grad_inner_fun(carry['inner_var'], x, start_inner),
            carry['outer_var']
        )
        implicit_grad = vjp_fun(v)[0]
        grad_outer_var = grad_out - implicit_grad

        # Step.4 - update the outer variables
        carry['outer_var'] -= outer_lr * grad_outer_var

        return carry, _

    init = dict(
        inner_var=inner_var, outer_var=outer_var, state_lr=state_lr,
        state_inner_sampler=state_inner_sampler,
        state_outer_sampler=state_outer_sampler,
        key=key
    )
    carry, _ = jax.lax.scan(
        ttsa_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )
    return carry['inner_var'], carry['outer_var'], \
        {k: v for k, v in carry.items()
         if k not in ['inner_var', 'outer_var']}
