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
    """Stochastic Recursive Bilevel Algorithm (SRBA).

    M. Dagréou, T. Moreau, S. Vaiter, P. Ablin. "A Lower Bound and a
    Near-Optimal Algorithmv for Bilevel Empirical Risk Minimizatio".
    arxiv:2302.08766 2023"""
    name = 'SRBA'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'batch_size': [64],
        'period_frac': [.5],
        'eval_freq': [128],
        'random_state': [1],
        'framework': ["jax"],
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
            # JIT necessary functions and classes
            self.srba = njit(srba)
            self.MinibatchSampler = jitclass(MinibatchSampler,
                                             mbs_spec)

            self.LearningRateScheduler = jitclass(LearningRateScheduler,
                                                  sched_spec)

        elif self.framework == 'none':
            self.f_inner = f_train(framework=self.framework)
            self.f_outer = f_val(framework=self.framework)
            self.srba = srba
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

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
            self.srba = partial(
                srba_jax,
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
        eval_freq = self.eval_freq  # // self.batch_size

        memory_start = get_memory()

        # Init variables
        inner_var = self.inner_var.copy()
        outer_var = self.outer_var.copy()

        if self.framework == "jax":
            v = jnp.zeros_like(inner_var)
            step_sizes = jnp.array(  # (inner_ss, hia_lr, outer_ss)
                [
                    self.step_size,
                    self.step_size / self.outer_ratio,
                ]
            )
            exponents = jnp.zeros(2)
            state_lr = init_lr_scheduler(step_sizes, exponents)
            d_inner = jnp.zeros_like(inner_var)
            d_v = jnp.zeros_like(inner_var)
            d_outer = jnp.zeros_like(outer_var)
            carry = dict(
                state_lr=state_lr,
                state_inner_sampler=self.state_inner_sampler,
                state_outer_sampler=self.state_outer_sampler,
                i_min=0
            )
        else:
            rng = np.random.RandomState(self.random_state)
            v = np.zeros_like(inner_var)

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
                    self.step_size / self.outer_ratio,
                ]
            )
            exponents = np.zeros(2)
            lr_scheduler = self.LearningRateScheduler(
                np.array(step_sizes, dtype=float), exponents
            )
            d_inner = np.zeros_like(inner_var)
            d_v = np.zeros_like(inner_var)
            d_outer = np.zeros_like(outer_var)

        period = self.n_inner_samples + self.n_outer_samples
        period *= self.period_frac
        period /= self.batch_size
        period = int(period)

        inner_var_old = inner_var.copy()
        outer_var_old = outer_var.copy()
        v_old = v.copy()
        i_min = 0

        # Start algorithm
        while callback():
            if self.framework == "jax":
                (inner_var, outer_var, v, inner_var_old, outer_var_old,
                 v_old, d_inner, d_v, d_outer, carry) = self.srba(
                        self.f_inner, self.f_outer, self.f_inner_fb,
                        self.f_outer_fb, inner_var, outer_var, v,
                        inner_var_old, outer_var_old, v_old, d_inner,
                        d_v, d_outer, period=period, max_iter=eval_freq,
                        **carry
                    )
            else:
                (inner_var, outer_var, v, inner_var_old, outer_var_old,
                 v_old, d_inner, d_v, d_outer, i_min) = self.srba(
                        self.f_inner, self.f_outer,
                        inner_var, outer_var, v,
                        inner_var_old=inner_var_old, v_old=v_old,
                        outer_var_old=outer_var_old, d_inner=d_inner, d_v=d_v,
                        d_outer=d_outer, inner_sampler=inner_sampler,
                        outer_sampler=outer_sampler, lr_scheduler=lr_scheduler,
                        i_min=i_min, period=period, max_iter=eval_freq,
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


def srba(
    inner_oracle, outer_oracle, inner_var, outer_var, v, inner_var_old,
    outer_var_old, v_old, d_inner, d_v, d_outer, inner_sampler=None,
    outer_sampler=None, lr_scheduler=None, i_min=0, period=100, max_iter=1,
    seed=None
):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(i_min, i_min+max_iter):
        inner_lr, outer_lr = lr_scheduler.get_lr()

        # Computation of the directions
        if i % period == 0:  # Full batch computations
            slice_inner = slice(0, inner_oracle.n_samples)
            _, d_inner, hvp, cross_v = inner_oracle.oracles(
                inner_var,
                outer_var,
                v,
                slice_inner,
                inverse='id'
            )

            slice_outer = slice(0, outer_oracle.n_samples)
            grad_outer_in, grad_outer_out = outer_oracle.grad(
                inner_var,
                outer_var,
                slice_outer
            )
            d_v = hvp + grad_outer_in
            d_outer = cross_v + grad_outer_out

        else:  # Stochastic computations
            slice_inner, _ = inner_sampler.get_batch()
            _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
                inner_var, outer_var, v, slice_inner, inverse='id'
            )
            _, grad_inner_var_old, hvp_old, cross_v_old = inner_oracle.oracles(
                inner_var_old, outer_var_old, v_old, slice_inner, inverse='id'
            )

            slice_outer, _ = outer_sampler.get_batch()
            grad_outer_in, grad_outer_out = outer_oracle.grad(
                inner_var, outer_var, slice_outer
            )
            grad_outer_in_old, grad_outer_out_old = outer_oracle.grad(
                inner_var_old, outer_var_old, slice_outer
            )

            d_inner += grad_inner_var - grad_inner_var_old
            d_v += (hvp - hvp_old) + (grad_outer_in - grad_outer_in_old)
            d_outer += (cross_v - cross_v_old)
            d_outer += (grad_outer_out - grad_outer_out_old)

        # Store the last iterates
        inner_var_old = inner_var.copy()
        v_old = v.copy()
        outer_var_old = outer_var.copy()
        # Update of the variables
        inner_var -= inner_lr * d_inner
        v -= inner_lr * d_v
        outer_var -= outer_lr * d_outer
    return (
        inner_var, outer_var, v, inner_var_old, outer_var_old, v_old, d_inner,
        d_v, d_outer, i_min+max_iter
    )


@partial(jax.jit, static_argnums=(0, 1, 2, 3),
         static_argnames=('inner_sampler', 'outer_sampler', 'period',
                          'max_iter'))
def srba_jax(f_inner, f_outer, f_inner_fb, f_outer_fb, inner_var, outer_var, v,
             inner_var_old, outer_var_old, v_old, d_inner, d_v, d_outer,
             state_inner_sampler=None, state_outer_sampler=None, state_lr=None,
             inner_sampler=None, outer_sampler=None, i_min=0, period=1,
             max_iter=1):
    def fb_directions(inner_var, outer_var, v, inner_var_old, outer_var_old,
                      v_old, d_inner, d_v, d_outer, state_inner_sampler,
                      state_outer_sampler):
        d_inner, vjp_train = jax.vjp(
            lambda z, x: jax.grad(f_inner_fb, argnums=0)(z, x),
            inner_var, outer_var
        )
        hvp, cross_v = vjp_train(v)
        grad_outer_in, grad_outer_out = jax.grad(
            f_outer_fb, argnums=(0, 1))(inner_var, outer_var)
        d_v = hvp + grad_outer_in
        d_outer = cross_v + grad_outer_out
        return d_inner, d_v, d_outer, state_inner_sampler, state_outer_sampler

    def srba_directions(inner_var, outer_var, v, inner_var_old, outer_var_old,
                        v_old, d_inner, d_v, d_outer, state_inner_sampler,
                        state_outer_sampler):
        start_inner, *_, state_inner_sampler = (
            inner_sampler(state_inner_sampler))
        start_outer, *_, state_outer_sampler = (
            outer_sampler(state_outer_sampler))
        grad_inner_var, vjp_train = jax.vjp(
            lambda z, x: jax.grad(f_inner, argnums=0)(z, x, start_inner),
            inner_var, outer_var
        )
        hvp, cross_v = vjp_train(v)
        grad_outer_in, grad_outer_out = jax.grad(f_outer, argnums=(0, 1))(
            inner_var, outer_var, start_outer
        )

        grad_inner_var_old, vjp_train_old = jax.vjp(
            lambda z, x: jax.grad(f_inner, argnums=0)(z, x, start_inner),
            inner_var_old, outer_var_old
        )
        hvp_old, cross_v_old = vjp_train_old(v_old)
        grad_outer_in_old, grad_outer_out_old = jax.grad(
            f_outer, argnums=(0, 1))(inner_var_old, outer_var_old, start_outer)

        d_inner += grad_inner_var - grad_inner_var_old
        d_v += (hvp - hvp_old) + (grad_outer_in - grad_outer_in_old)
        d_outer += (cross_v - cross_v_old)
        d_outer += (grad_outer_out - grad_outer_out_old)

        return d_inner, d_v, d_outer, state_inner_sampler, state_outer_sampler

    def srba_one_iter(carry, i):
        (inner_lr, outer_lr), carry['state_lr'] = update_lr(carry['state_lr'])
        carry['d_inner'], carry['d_v'], carry['d_outer'], \
            carry['state_inner_sampler'], carry['state_outer_sampler'] = \
            jax.lax.cond(
                i % period == 0, fb_directions, srba_directions,
                carry['inner_var'], carry['outer_var'], carry['v'],
                carry['inner_var_old'], carry['outer_var_old'],
                carry['v_old'], carry['d_inner'], carry['d_v'],
                carry['d_outer'], carry['state_inner_sampler'],
                carry['state_outer_sampler']
            )

        carry['inner_var_old'] = carry['inner_var'].copy()
        carry['v_old'] = carry['v'].copy()
        carry['outer_var_old'] = carry['outer_var'].copy()

        # Update of the variables
        carry['inner_var'] -= inner_lr * carry['d_inner']
        carry['v'] -= inner_lr * carry['d_v']
        carry['outer_var'] -= outer_lr * carry['d_outer']

        # #Use prox to make sure we do not diverge
        # # inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)
        return carry, None

    init = dict(
        inner_var=inner_var, outer_var=outer_var, v=v,
        inner_var_old=inner_var_old, outer_var_old=outer_var_old, v_old=v_old,
        d_inner=d_inner, d_v=d_v, d_outer=d_outer, state_lr=state_lr,
        state_inner_sampler=state_inner_sampler,
        state_outer_sampler=state_outer_sampler,
        i_min=i_min
    )
    carry, _ = jax.lax.scan(
        srba_one_iter,
        init=init,
        xs=jnp.arange(0, max_iter) + init['i_min'],
        length=max_iter,
    )
    carry['i_min'] += max_iter
    return (
        carry['inner_var'], carry['outer_var'], carry['v'],
        carry['inner_var_old'], carry['outer_var_old'], carry['v_old'],
        carry['d_inner'], carry['d_v'], carry['d_outer'],
        {k: v for k, v in carry.items()
         if k not in ['inner_var', 'outer_var', 'v',
                      'inner_var_old', 'outer_var_old', 'v_old',
                      'd_inner', 'd_v', 'd_outer']}
    )
