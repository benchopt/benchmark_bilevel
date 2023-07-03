from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils import constants
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp
    from functools import partial

    import jaxopt


class Solver(BaseSolver):
    """Two loops solver."""
    name = 'jaxopt ITD'

    requirements = ["pip:jaxopt"]

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'inner_solver': ['gd'],
        'step_size_outer': [1.],
        'eval_freq': [1],
        'n_inner_steps': [100],
        'warm_start': [True, False]
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_train, f_val, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):
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

        @partial(jax.jit, static_argnames=("f", "n_steps"))
        def inner_solver_fun(outer_var, inner_var, f=None, n_steps=1, lr=.1):
            if self.inner_solver == 'gd':
                solver = jaxopt.GradientDescent(
                    fun=f, maxiter=n_steps, implicit_diff=False,
                    acceleration=False, unroll=True
                )
            else:
                raise ValueError(f"Inner solver {self.inner_solver} not" 
                                 + "available")
            return solver.run(inner_var, outer_var).params

        self.inner_solver_fun = partial(inner_solver_fun,
                                        f=self.f_inner,
                                        n_steps=self.n_inner_steps)
        self.jaxopt_solver = partial(jaxopt_bilevel_solver,
                                     inner_solver=self.inner_solver_fun)

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.run_once(2)

    def run(self, callback):
        eval_freq = self.eval_freq

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()

        step_sizes = jnp.array(
            [self.step_size_outer]
        )
        exponents = jnp.zeros(1)
        state_lr = init_lr_scheduler(step_sizes, exponents)

        carry = dict(
            state_lr=state_lr,
        )

        while callback((inner_var, outer_var)):
            inner_var, outer_var, carry = self.jaxopt_solver(
                    self.f_inner, self.f_outer, inner_var, outer_var,
                    n_inner_steps=self.n_inner_steps, max_iter=eval_freq,
                    warm_start=self.warm_start, inner_var_0=self.inner_var0,
                    **carry
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


@partial(jax.jit, static_argnums=(0, 1),
         static_argnames=('n_inner_steps', 'max_iter', "inner_solver",
                          "warm_start"))
def jaxopt_bilevel_solver(f_inner, f_outer, inner_var, outer_var,
                          state_lr=None, n_inner_steps=300,
                          inner_solver=None, inner_var_0=None, warm_start=True,
                          max_iter=1):
    grad_outer = jax.grad(f_outer, argnums=(0, 1))

    def jaxopt_one_iter(carry, _):
        outer_lr, carry['state_lr'] = update_lr(
            carry['state_lr']
        )
        init_inner = carry['inner_var'] if warm_start else inner_var_0
        carry['inner_var'], jvp_fun = jax.vjp(inner_solver,
                                              carry['outer_var'],
                                              init_inner)

        grad_outer_in, grad_outer_out = grad_outer(carry['inner_var'],
                                                   carry['outer_var'])

        implicit_grad = grad_outer_out + jvp_fun(grad_outer_in)[0]
        carry['outer_var'] -= outer_lr * implicit_grad

        return carry, _

    init = dict(
        inner_var=inner_var, outer_var=outer_var, state_lr=state_lr
    )
    carry, _ = jax.lax.scan(
        jaxopt_one_iter,
        init=init,
        xs=None,
        length=max_iter,
    )
    return carry['inner_var'], carry['outer_var'], \
        {k: v for k, v in carry.items()
         if k not in ['inner_var', 'outer_var']}
