from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import jax
    import optuna
    import jax.numpy as jnp

    from jaxopt import LBFGS
    from functools import partial


class Solver(BaseSolver):
    """Hyperparameter Selection with Optuna.

    T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama. "Optuna: A
    Next-generation Hyperparameter Optimization Framework". KDD 2019."""
    name = 'Optuna'
    stopping_criterion = SufficientProgressCriterion(
        patience=100, strategy='callback'
    )

    install_cmd = 'conda'
    requirements = ['pip::optuna', 'pip::jaxopt']
    parameters = {
        'random_state': [1],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_inner, f_outer, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        self.f_inner = partial(f_inner, start=0, batch_size=n_inner_samples)
        self.f_outer = partial(f_outer, start=0, batch_size=n_outer_samples)

        self.solver_inner = LBFGS(fun=self.f_inner)

        @jax.jit
        def value_function(outer_var):
            inner_var = self.solver_inner.run(
                self.inner_var0, outer_var
            ).params
            return self.f_outer(inner_var, outer_var), inner_var

        def obj_optuna(trial):
            outer_var_flat = jnp.empty_like(self.outer_var0.ravel())
            for k in range(len(outer_var_flat)):
                outer_var_flat = outer_var_flat.at[k].set(
                    trial.suggest_float(f'outer_var_{k}', -5, 5)
                )

            outer_var = outer_var_flat.reshape(self.outer_var.shape)
            loss, inner_var = value_function(outer_var)
            trial.set_user_attr('inner_var', inner_var)
            return loss
        self.obj_optuna = obj_optuna

        self.run_once(2)

    def run(self, cb):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.outer_var = self.outer_var0.copy()
        self.inner_var = self.inner_var0.copy()

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(sampler=sampler)
        study.enqueue_trial({
            f"outer_var_{k}": v for k, v in enumerate(self.outer_var0.ravel())
        })

        while cb():
            study.optimize(self.obj_optuna, n_trials=5)
            trial = study.best_trial
            self.outer_var = jnp.array(list(trial.params.values())).reshape(
                self.outer_var.shape
            )
            self.inner_var = trial.user_attrs['inner_var']

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var)
