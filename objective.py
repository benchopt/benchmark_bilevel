from benchopt import BaseObjective
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.utils import check_random_state

    oracles = import_ctx.import_from("oracles")


class Objective(BaseObjective):
    name = "Bilevel"

    parameters = {
        'task': ['datacleaning', 'multilogreg']
    }

    def __init__(self, task='datacleaning', random_state=2442):
        if task == 'datacleaning':
            self.get_inner_oracle = oracles.DataCleaningOracle
            self.get_outer_oracle = oracles.MultinomialLogRegOracle
        elif task == 'multilogreg':
            self.get_inner_oracle = oracles.MultiLogRegOracle
            self.get_outer_oracle = (
                lambda X, y: oracles.MultiLogRegOracle(X, y, reg='none')
            )
        else:
            raise ValueError

        self.random_state = random_state

    def set_data(self, X_train, y_train, X_test, y_test, X_val, y_val):
        self.f_train = self.get_inner_oracle(X_train, y_train)
        self.f_test = self.get_outer_oracle(X_test, y_test)
        self.X_val = X_val
        self.y_val = y_val

        # Init inner and outer variables
        # rng = check_random_state(self.random_state)
        inner_shape, outer_shape = self.f_train.variables_shape
        # self.inner_var0 = rng.randn(*inner_shape)
        self.inner_var0 = np.zeros(*inner_shape)
        self.outer_var0 = np.ones(*outer_shape)
        self.inner_var0, self.outer_var0 = self.f_train.prox(
            self.inner_var0, self.outer_var0
        )

    def compute(self, beta):
        inner_var, outer_var = beta

        if np.isnan(outer_var).any():
            raise ValueError
        acc = self.f_test.accuracy(inner_var, outer_var, self.X_val, self.y_val)
        # inner_star = self.f_train.get_inner_var_star(outer_var)
        # value_function = self.f_test.get_value(inner_star, outer_var)
        # inner_value = self.f_train.get_value(inner_var, outer_var)
        # outer_value = self.f_test.get_value(inner_var, outer_var)
        # d_inner = np.linalg.norm(inner_var - inner_star)
        # d_value = outer_value - value_function
        # grad_f_test_inner, grad_f_test_outer = self.f_test.get_grad(
        #     inner_star, outer_var
        # )
        # grad_value = grad_f_test_outer
        # v = self.f_train.get_inverse_hvp(
        #     inner_star, outer_var,
        #     grad_f_test_inner
        # )
        # grad_value -= self.f_train.get_cross(inner_star, outer_var, v)

        return dict(value=acc)

    def to_dict(self):
        return dict(
            f_train=self.f_train,
            f_test=self.f_test,
            inner_var0=self.inner_var0,
            outer_var0=self.outer_var0,
        )
