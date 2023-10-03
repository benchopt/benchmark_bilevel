from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.get_memory import get_memory
    from benchmark_utils.sgd_inner import sgd_inner_vrbo
    from benchmark_utils.sgd_inner import sgd_inner_vrbo_jax
    from benchmark_utils.minibatch_sampler import init_sampler
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.hessian_approximation import shia_fb_jax
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.hessian_approximation import joint_shia_jax
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler
    from benchmark_utils.hessian_approximation import shia_fb, joint_shia
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler
    from benchmark_utils.oracles import MultiLogRegOracle, DataCleaningOracle

    import jax
    import jax.numpy as jnp
    from functools import partial


class Solver(BaseSolver):
    """Variance Reduction Bilevel Optimizer (VRBO).

    J. Yang, K. Ji, Y. Liang. "Provabily Faster Algorithms for Bilevel
    Optimization". NeurIPS 2021"""
    name = 'VRBO'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_shia_steps': [10],
        'batch_size': [64],
        'period_frac': [128],
        'eval_freq': [128],
        'n_inner_steps': [10],
        'random_state': [1],
        'framework': ['jax']
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

        self.n_inner_samples = n_inner_samples
        self.n_outer_samples = n_outer_samples

        if self.batch_size == "full":
            self.batch_size_inner = n_inner_samples
            self.batch_size_outer = n_outer_samples
        else:
            self.batch_size_inner = self.batch_size
            self.batch_size_outer = self.batch_size

        if self.framework == 'numba':
            self.f_inner = f_train(framework=self.framework)
            self.f_outer = f_val(framework=self.framework)
            njit_vrbo = njit(_vrbo)
            njit_shia = njit(shia_fb)
            njit_joint_shia = njit(joint_shia)
            _sgd_inner_vrbo = njit(sgd_inner_vrbo)

            @njit
            def njit_sgd_inner_vrbo(
                inner_oracle, outer_oracle,  inner_var,  outer_var, inner_lr,
                inner_sampler, outer_sampler, n_inner_steps, memory_inner,
                memory_outer, n_shia_steps, hia_lr
            ):
                return _sgd_inner_vrbo(
                    njit_joint_shia, inner_oracle, outer_oracle, inner_var,
                    outer_var, inner_lr, inner_sampler, outer_sampler,
                    n_inner_steps, memory_inner, memory_outer, n_shia_steps,
                    hia_lr
                )

            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

            def vrbo(*args, **kwargs):
                return njit_vrbo(njit_sgd_inner_vrbo, njit_shia, *args,
                                 **kwargs)
            self.vrbo = vrbo

        elif self.framework == 'none':
            self.f_inner = f_train(framework=self.framework)
            self.f_outer = f_val(framework=self.framework)

            def _sgd_inner_vrbo(*args, **kwargs):
                return sgd_inner_vrbo(joint_shia, *args, *kwargs)
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

            def vrbo(*args, **kwargs):
                return _vrbo(_sgd_inner_vrbo, shia_fb, *args, **kwargs)
            self.vrbo = vrbo
        elif self.framework == 'jax':
            self.f_inner, self.f_inner_fb = f_train(
                framework=self.framework, get_full_batch=True
            )
            self.f_outer, self.f_outer_fb = f_val(
                framework=self.framework, get_full_batch=True
            )
            self.f_inner = jax.jit(
                partial(self.f_inner, batch_size=self.batch_size_inner)
            )
            self.f_outer = jax.jit(
                partial(self.f_outer, batch_size=self.batch_size_outer)
            )
            self.f_inner_fb = jax.jit(self.f_inner_fb)
            self.f_outer_fb = jax.jit(self.f_outer_fb)
            inner_sampler, self.state_inner_sampler \
                = init_sampler(n_samples=n_inner_samples,
                               batch_size=self.batch_size_inner)
            outer_sampler, self.state_outer_sampler \
                = init_sampler(n_samples=n_outer_samples,
                               batch_size=self.batch_size_outer)
            self.sgd_inner = partial(sgd_inner_vrbo_jax,
                                     joint_shia=joint_shia_jax,
                                     inner_sampler=inner_sampler,
                                     outer_sampler=outer_sampler,
                                     grad_inner_fun=jax.grad(self.f_inner,
                                                             argnums=0),
                                     grad_outer_fun=jax.grad(self.f_outer,
                                                             argnums=(0, 1)))

            self.vrbo = partial(
                vrbo_jax,
                shia=shia_fb_jax,
                sgd_inner_vrbo=self.sgd_inner,
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
        memory_start = get_memory()

        # Init variables
        inner_var = self.inner_var.copy()
        outer_var = self.outer_var.copy()

        period = self.n_inner_samples + self.n_outer_samples
        period *= self.period_frac
        period /= self.batch_size
        period = int(period)

        if self.framework == 'jax':
            inner_var_old = jnp.zeros_like(inner_var)
            d_inner = jnp.zeros_like(inner_var)
            d_outer = jnp.zeros_like(outer_var)
            step_sizes = jnp.array(  # (inner_ss, hia_lr, outer_ss)
                [
                    self.step_size,
                    self.step_size,
                    self.step_size / self.outer_ratio,
                ]
            )
            exponents = jnp.zeros(3)
            state_lr = init_lr_scheduler(step_sizes, exponents)
            carry = dict(
                state_lr=state_lr,
                state_inner_sampler=self.state_inner_sampler,
                state_outer_sampler=self.state_outer_sampler,
                i_min=0
            )
        else:
            rng = np.random.RandomState(self.random_state)
            memory_inner = np.zeros((2, *inner_var.shape), inner_var.dtype)
            memory_outer = np.zeros((2, *outer_var.shape), outer_var.dtype)

            # Init sampler and lr scheduler
            inner_sampler = self.MinibatchSampler(
                self.f_inner.n_samples, batch_size=self.batch_size_inner
            )
            outer_sampler = self.MinibatchSampler(
                self.f_outer.n_samples, batch_size=self.batch_size_outer
            )
            step_sizes = np.array(  # (inner_ss, hia_lr, outer_ss)
                [
                    self.step_size,
                    self.step_size,
                    self.step_size / self.outer_ratio,
                ]
            )
            exponents = np.zeros(3)
            lr_scheduler = self.LearningRateScheduler(
                np.array(step_sizes, dtype=float), exponents
            )
        i_min = 0
        # Start algorithm
        while callback():
            if self.framework == 'jax':
                inner_var, outer_var, inner_var_old, d_inner, d_outer, \
                    carry = self.vrbo(
                                self.f_inner, self.f_outer, self.f_inner_fb,
                                self.f_outer_fb, inner_var, outer_var,
                                inner_var_old, d_inner, d_outer,
                                n_inner_steps=self.n_inner_steps,
                                n_shia_steps=self.n_shia_steps,
                                max_iter=eval_freq, period=period, **carry
                            )
            else:
                inner_var, outer_var, memory_inner, memory_outer, i_min = \
                    self.vrbo(
                        self.f_inner, self.f_outer,
                        inner_var, outer_var, memory_inner, memory_outer,
                        eval_freq, inner_sampler, outer_sampler,
                        lr_scheduler, self.n_shia_steps, self.n_inner_steps,
                        i_min=i_min, period=period,
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


def _vrbo(
    sgd_inner_vrbo, shia, inner_oracle, outer_oracle, inner_var,
    outer_var, memory_inner, memory_outer, max_iter, inner_sampler,
    outer_sampler, lr_scheduler, n_shia_steps, n_inner_steps, i_min=0,
    period=100, seed=None
):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(i_min, i_min+max_iter):
        inner_lr, hia_lr, outer_lr = lr_scheduler.get_lr()
        # outer_lr = 0.

        # Step.1 - (Re)initialize directions for z and x
        if i % period == 0:
            slice_inner = slice(0, inner_oracle.n_samples)
            grad_inner_var = inner_oracle.grad_inner_var(
                inner_var, outer_var, slice_inner
            )
            memory_inner[1] = grad_inner_var

            slice_outer = slice(0, outer_oracle.n_samples)
            grad_outer, impl_grad = outer_oracle.grad(
                inner_var, outer_var, slice_outer
            )
            ihvp = shia(
                inner_oracle, inner_var, outer_var, grad_outer, n_shia_steps,
                hia_lr
            )
            impl_grad -= inner_oracle.cross(
                inner_var, outer_var, ihvp, slice_inner
            )
            memory_outer[1] = impl_grad

        # Step.2 - Update outer variable
        outer_var -= outer_lr * memory_outer[1]

        inner_var, outer_var, memory_inner, memory_outer = sgd_inner_vrbo(
            inner_oracle, outer_oracle, inner_var, outer_var, inner_lr,
            inner_sampler, outer_sampler, n_inner_steps, memory_inner,
            memory_outer, n_shia_steps, hia_lr
        )

    return inner_var, outer_var, memory_inner, memory_outer, i_min+max_iter


@partial(jax.jit, static_argnums=(0, 1, 2, 3),
         static_argnames=('inner_sampler', 'outer_sampler', 'period',
                          'max_iter', 'n_inner_steps', 'n_shia_steps', 'shia',
                          'sgd_inner_vrbo'))
def vrbo_jax(f_inner, f_outer, f_inner_fb, f_outer_fb, inner_var, outer_var,
             inner_var_old, d_inner, d_outer, n_shia_steps=1, i_min=0,
             period=1, sgd_inner_vrbo=None, n_inner_steps=1,
             state_lr=None, state_inner_sampler=None,
             state_outer_sampler=None, inner_sampler=None,
             outer_sampler=None, shia=None, max_iter=1):
    def fb_directions(inner_var, outer_var, hia_lr, d_inner, d_outer):
        grad_inner_fun = jax.grad(f_inner_fb, argnums=0)
        grad_inner, cross_v = jax.vjp(
            lambda x: grad_inner_fun(inner_var, x), outer_var
        )
        grad_outer_in, grad_outer_out = jax.grad(f_outer_fb, argnums=(0, 1))(
            inner_var, outer_var
        )
        v = shia(
            inner_var, outer_var, grad_outer_in, hia_lr,
            n_steps=n_shia_steps, grad_inner=grad_inner_fun
        )
        d_inner = grad_inner
        d_outer = grad_outer_out - cross_v(v)[0]
        return d_inner, d_outer

    def identity_directions(inner_var, outer_var, hia_lr, d_inner, d_outer):
        return d_inner, d_outer

    def vrbo_one_iter(carry, i):
        (inner_lr, hia_lr, outer_lr), carry['state_lr'] = update_lr(
            carry['state_lr']
        )

        # Step.1 - (Re)initialize directions for z and x
        carry['d_inner'], carry['d_outer'] = jax.lax.cond(
            i % period == 0, fb_directions, identity_directions,
            carry['inner_var'], carry['outer_var'], hia_lr, carry['d_inner'],
            carry['d_outer']
        )
        # Step.2 - Update outer variable
        carry['outer_var'] -= outer_lr * carry['d_outer']

        carry['inner_var'], carry['inner_var_old'], carry['d_inner'], \
            carry['d_outer'], carry['state_inner_sampler'], \
            carry['state_outer_sampler'] = sgd_inner_vrbo(
                carry['inner_var'], carry['outer_var'], carry['inner_var_old'],
                carry['d_inner'], carry['d_outer'],
                carry['state_inner_sampler'], carry['state_outer_sampler'],
                inner_lr, hia_lr, n_shia_steps=n_shia_steps,
                n_steps=n_inner_steps
            )

        return carry, None

    init = dict(
        inner_var=inner_var, outer_var=outer_var, inner_var_old=inner_var_old,
        d_inner=d_inner, d_outer=d_outer, state_lr=state_lr,
        state_inner_sampler=state_inner_sampler,
        state_outer_sampler=state_outer_sampler, i_min=i_min
    )
    carry, _ = jax.lax.scan(
        vrbo_one_iter,
        init=init,
        xs=jnp.arange(0, max_iter) + init['i_min'],
        length=max_iter,
    )
    carry['i_min'] += max_iter
    return (
        carry['inner_var'], carry['outer_var'], carry['inner_var_old'],
        carry['d_inner'], carry['d_outer'],
        {k: v for k, v in carry.items()
         if k not in ['inner_var', 'outer_var', 'inner_var_old',
                      'd_inner', 'd_outer']}
    )
