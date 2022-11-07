from benchopt import BaseSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import optuna
    constants = import_ctx.import_from('constants')

class Solver(BaseSolver):
    """Hyperparameter Selection with Optuna."""
    name = 'Optuna'
    stopping_strategy = 'iteration'

    install_cmd = 'conda'
    requirements = ['pip:optuna']

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_train, f_test, inner_var0, outer_var0, numba):
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        self.f_inner = f_train
        self.f_outer = f_test
        self.numba = numba

    def run(self, n_iter):
        if n_iter == 0:
            inner_var = self.f_inner.get_inner_var_star(outer_var)
            outer_var = self.outer_var0.copy()
        else:
            def obj_optuna(trial):
                outer_var_flat = self.outer_var0.ravel()
                for k in range(len(outer_var_flat)):
                    outer_var_flat[k] = trial.suggest_float(
                        f'outer_var{k}',
                        -5,
                        5
                    )
                inner_var = self.f_inner.get_inner_var_star(outer_var)
                return self.f_outer.get_value(inner_var, outer_var)

            study = optuna.create_study(direction='minimize')
            study.optimize(obj_optuna, n_trials=n_iter)
            trial = study.best_trial
            outer_var = np.array(list(trial.params.values())).reshape(
                self.outer_var0.shape
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta
