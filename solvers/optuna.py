from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    import optuna


class Solver(BaseSolver):
    """Hyperparameter Selection with Optuna."""
    name = 'Optuna'
    stopping_criterion = SufficientProgressCriterion(
        patience=100, strategy='iteration'
    )

    install_cmd = 'conda'
    requirements = ['pip:optuna']
    parameters = {
        'random_state': [1],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_train, f_val, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        self.f_inner = f_train(framework='none')
        self.f_outer = f_val(framework='none')

    def run(self, n_iter):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if n_iter == 0:
            outer_var = self.outer_var0.copy()
        else:
            def obj_optuna(trial):
                outer_var_flat = self.outer_var0.ravel()
                for k in range(len(outer_var_flat)):
                    outer_var_flat[k] = trial.suggest_float(
                        f'outer_var{k}',
                        -15,
                        5
                    )
                outer_var = outer_var_flat.reshape(self.outer_var0.shape)
                inner_var = self.f_inner.get_inner_var_star(outer_var)
                return self.f_outer.get_value(inner_var, outer_var)

            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(obj_optuna, n_trials=n_iter)
            trial = study.best_trial
            outer_var = np.array(list(trial.params.values())).reshape(
                self.outer_var0.shape
            )
        inner_var = self.f_inner.get_inner_var_star(outer_var)
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta
