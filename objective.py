from benchopt import BaseObjective

import numpy as np
from sklearn.utils import check_random_state

import sys
from pathlib import Path
oracle_module = Path(__file__).parent
sys.path += [str(oracle_module.resolve())]

from oracles import RidgeRegressionOracle  # noqa: E402
from oracles import LogisticRegressionOracle  # noqa: E402


class Objective(BaseObjective):
    name = "Bi-level Hyperparameter Optimization"

    parameters = {
        'model': ['logreg', 'ridge']
    }

    def __init__(self, model='ridge', random_state=29):
        if model == 'ridge':
            self.oracle = RidgeRegressionOracle
        elif model == 'logreg':
            self.oracle = LogisticRegressionOracle
        else:
            raise ValueError(
                f"model should be 'ridge' or 'logreg'. Got '{model}'."
            )

        self.random_state = random_state

    def set_data(self, X_train, y_train, X_test, y_test):
        self.f_train = self.oracle(
            X_train, y_train, reg='lin'
        )
        self.f_test = self.oracle(
            X_test, y_test, reg='none'
        )

        rng = check_random_state(self.random_state)
        inner_shape, outer_shape = self.f_train.variables_shape
        self.inner_var0 = rng.randn(*inner_shape)
        # self.outer_var0 = np.log(2 * rng.rand(*outer_shape))
        self.outer_var0 = 10 * rng.rand(*outer_shape)
        self.inner_var0, self.outer_var0 = self.f_train.prox(
            self.inner_var0, self.outer_var0
        )

    def compute(self, beta):
        inner_var, outer_var = beta

        id_inner = np.arange(self.f_train.n_samples)
        id_outer = np.arange(self.f_test.n_samples)
        inner_star = self.f_train.inner_var_star(outer_var, id_inner)
        value_function = self.f_test.value(inner_star, outer_var, id_outer)
        inner_value = self.f_train.value(inner_var, outer_var, id_inner)
        outer_value = self.f_test.value(inner_var, outer_var, id_outer)
        d_inner = np.linalg.norm(inner_var - inner_star)
        d_value = outer_value - value_function

        return dict(
            value=value_function,
            inner_value=inner_value,
            outer_value=outer_value,
            d_inner=d_inner,
            d_value=d_value
        )

    def to_dict(self):
        return dict(
            f_train=self.f_train,
            f_test=self.f_test,
            inner_var0=self.inner_var0,
            outer_var0=self.outer_var0
        )
