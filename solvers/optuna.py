from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import optuna
    import numpy as np
    from benchmark_utils.get_memory import get_memory


class Solver(BaseSolver):
    """Hyperparameter Selection with Optuna.

    T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama. "Optuna: A
    Next-generation Hyperparameter Optimization Framework". KDD 2019."""
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
        self.inner_var = inner_var0
        self.outer_var = outer_var0

        self.f_inner = f_train(framework='none')
        self.f_outer = f_val(framework='none')

    def run(self, n_iter):
        memory_start = get_memory()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if n_iter == 0:
            outer_var = self.outer_var.copy()
        else:
            def obj_optuna(trial):
                outer_var_flat = self.outer_var.ravel()
                for k in range(len(outer_var_flat)):
                    outer_var_flat[k] = trial.suggest_float(
                        f'outer_var{k}',
                        -15,
                        5
                    )
                outer_var = outer_var_flat.reshape(self.outer_var.shape)
                inner_var = self.f_inner.inner_var_star(outer_var)
                return self.f_outer.get_value(inner_var, outer_var)

            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(obj_optuna, n_trials=n_iter)
            trial = study.best_trial
            outer_var = np.array(list(trial.params.values())).reshape(
                self.outer_var.shape
            )

        memory_end = get_memory()
        self.inner_var = self.f_inner.inner_var_star(outer_var)
        self.outer_var = outer_var
        self.memory = memory_end - memory_start
        self.memory /= 1e6

    def get_result(self):
        return dict(inner_var=self.inner_var, outer_var=self.outer_var,
                    memory=self.memory)
