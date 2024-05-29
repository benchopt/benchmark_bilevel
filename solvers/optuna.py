from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import jax
    import optuna
    import jax.numpy as jnp

    from jaxopt import LBFGS


class Solver(BaseSolver):
    """Hyperparameter Selection with Optuna.

    T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama. "Optuna: A
    Next-generation Hyperparameter Optimization Framework". KDD 2019."""
    name = 'Optuna'
    stopping_criterion = SufficientProgressCriterion(
        patience=100, strategy='iteration'
    )

    install_cmd = 'conda'
    requirements = ['pip:optuna', 'pip:jaxopt']
    parameters = {
        'random_state': [1],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_inner, f_outer, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0, f_inner_fb, f_outer_fb,):
        self.inner_var = inner_var0
        self.outer_var = outer_var0

        self.f_inner = f_inner_fb
        self.f_outer = f_outer_fb

        self.solver_inner = LBFGS(fun=self.f_inner)

        @jax.jit
        def get_inner_sol(inner_var_init, outer_var):
            return self.solver_inner.run(inner_var_init,
                                         outer_var).params
        self.get_inner_sol = get_inner_sol

        self.run_once(2)

    def run(self, n_iter):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if n_iter == 0:
            outer_var = self.outer_var.copy()
        else:
            def obj_optuna(trial):
                outer_var_flat = self.outer_var.ravel()
                for k in range(len(outer_var_flat)):
                    outer_var_flat.at[k].set(
                        trial.suggest_float(
                            f'outer_var{k}',
                            -15,
                            5
                        )
                    )
                outer_var = outer_var_flat.reshape(self.outer_var.shape)
                inner_var = self.get_inner_sol(self.inner_var, self.outer_var)
                return self.f_outer(inner_var, outer_var)

            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(obj_optuna, n_trials=n_iter)
            trial = study.best_trial
            outer_var = jnp.array(list(trial.params.values())).reshape(
                self.outer_var.shape
            )

        self.outer_var = outer_var
        self.inner_var = self.get_inner_sol(self.inner_var,
                                            self.outer_var)

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var)
