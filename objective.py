from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.utils import check_random_state
    # oracles = import_ctx.import_from('oracles')
    scipy_to_csrmatrix = import_ctx.import_from(
        'sparse_matrix', 'scipy_to_csrmatrix'
    )
    scipy_to_cscmatrix = import_ctx.import_from(
        'sparse_matrix', 'scipy_to_cscmatrix'
    )

oracles = safe_import_context().import_from('oracles')


class Objective(BaseObjective):
    name = "Bi-level Hyperparameter Optimization"

    is_convex = False

    def __init__(self, model='ridge', reg='exp',  n_reg='full',
                 random_state=2442):

        self.oracle = oracles.MulticlassLogisticRegressionOracle

        self.random_state = random_state

    def set_data(self, X_train, y_train, X_test, y_test):
        self.f_train = self.oracle(
            X_train, y_train, reg=True
        )
        self.f_test = self.oracle(
            X_test, y_test, reg=False
        )

        rng = check_random_state(self.random_state)
        inner_shape, outer_shape = self.f_train.variables_shape
        self.inner_var0 = rng.randn(*inner_shape[0]).ravel()
        self.outer_var0 = np.log(rng.rand(*outer_shape))

    def compute(self, beta):

        inner_var, outer_var = beta

        if np.isnan(outer_var).any():
            raise ValueError

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
        # grad_inner = self.f_train.get_grad_inner_var(inner_var, outer_var)
        # grad_star = self.f_train.get_grad_inner_var(inner_star, outer_var)

        # return dict(
        #     value_func=value_function,
        #     inner_value=inner_value,
        #     outer_value=outer_value,
        #     d_inner=d_inner,
        #     d_value=d_value,
        #     # value=np.linalg.norm(grad_value)**2,
        #     grad_inner=np.linalg.norm(grad_inner),
        #     grad_star=np.linalg.norm(grad_star),
        # )
        return dict(
            train_loss=self.f_train.get_value(inner_var, outer_var),
            value=self.f_test.get_value(inner_var, outer_var),
            norm_inner=np.linalg.norm(inner_var)
        )

    def to_dict(self):
        return dict(
            f_train=self.f_train,
            f_test=self.f_test,
            inner_var0=self.inner_var0,
            outer_var0=self.outer_var0
        )
