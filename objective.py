from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.utils import check_random_state

oracles = import_ctx.import_from('oracles')


class Objective(BaseObjective):
    name = "Bi-level Hyperparameter Optimization"

    parameters = {
        'model': ['logreg', 'ridge', 'quadratic'],
        'n_reg': [1, 'full'],
        'reg': ['exp', 'lin']
    }

    is_convex = False

    def __init__(self, model='ridge', reg='exp',  n_reg='full',
                 random_state=2442):
        if model == 'ridge':
            self.oracle = oracles.RidgeRegressionOracle
        elif model == 'logreg':
            self.oracle = oracles.LogisticRegressionOracle
        elif model == 'quadratic':
            self.oracle = oracles.QuadraticOracle
        else:
            raise ValueError(
                f"model should be 'ridge', 'logreg' or 'quadratic'. \
                    Got '{model}'."
            )

        self.reg = reg
        self.n_reg = n_reg
        self.random_state = random_state
        self.model = model

    def set_data(self, **data):
        rng = check_random_state(self.random_state)
        if self.model == 'quadratic':
            A_z_inner, A_x_inner, b_inner, c_inner = data['A_z_inner'], \
                data['A_x_inner'], data['b_inner'], data['c_inner']
            A_z_outer, A_x_outer, b_outer, c_outer = data['A_z_outer'], \
                data['A_x_outer'], data['b_outer'], data['c_outer']
            self.f_train = self.oracle(A_z_inner, A_x_inner, b_inner, c_inner)
            self.f_test = self.oracle(A_z_outer, A_x_outer, b_outer, c_outer)
            self.inner_var0 = rng.randn(A_z_inner.shape[0])
            self.outer_var0 = rng.randn(A_x_inner.shape[0])
        else:
            X_train, y_train, X_test, y_test = data['X_train'], \
                data['y_train'], data['X_test'], data['y_test']
            self.f_train = self.oracle(
                X_train, y_train, reg=self.reg
            )
            self.f_test = self.oracle(
                X_test, y_test, reg='none'
            )
            inner_shape, outer_shape = self.f_train.variables_shape
            self.inner_var0 = rng.randn(*inner_shape)
            self.outer_var0 = rng.rand(*outer_shape)
            if self.reg == 'exp':
                self.outer_var0 = np.log(self.outer_var0)
            if self.n_reg == 1:
                self.outer_var0 = self.outer_var0[:1]
            self.inner_var0, self.outer_var0 = self.f_train.prox(
                self.inner_var0, self.outer_var0
            )

    def compute(self, beta):

        inner_var, outer_var = beta

        if np.isnan(outer_var).any():
            raise ValueError

        inner_star = self.f_train.get_inner_var_star(outer_var)
        value_function = self.f_test.get_value(inner_star, outer_var)
        inner_value = self.f_train.get_value(inner_var, outer_var)
        outer_value = self.f_test.get_value(inner_var, outer_var)
        d_inner = np.linalg.norm(inner_var - inner_star)
        d_value = outer_value - value_function
        grad_f_test_inner, grad_f_test_outer = self.f_test.get_grad(
            inner_star, outer_var
        )
        grad_value = grad_f_test_outer
        v = self.f_train.get_inverse_hvp(
            inner_star, outer_var,
            grad_f_test_inner
        )
        grad_value -= self.f_train.get_cross(inner_star, outer_var, v)
        grad_inner = self.f_train.get_grad_inner_var(inner_var, outer_var)
        grad_star = self.f_train.get_grad_inner_var(inner_star, outer_var)

        return dict(
            value=value_function,
            inner_value=inner_value,
            outer_value=outer_value,
        )

    def to_dict(self):
        return dict(
            f_train=self.f_train,
            f_test=self.f_test,
            inner_var0=self.inner_var0,
            outer_var0=self.outer_var0
        )
