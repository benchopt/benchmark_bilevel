from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils import oracles
    from libsvmdata import fetch_libsvm
    from benchmark_utils.oracle_utils import convert_array_framework


class Dataset(BaseDataset):

    name = "ijcnn1"

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata', 'scikit-learn']

    parameters = {
        'reg': ['exp'],
        'n_reg': ['full'],
        'oracle': ['logreg'],
    }

    def get_data(self):
        X_train, y_train = fetch_libsvm('ijcnn1')
        X_val, y_val = fetch_libsvm('ijcnn1_test')

        def get_inner_oracle(framework="none", get_full_batch=False):
            X = convert_array_framework(X_train, framework)
            y = convert_array_framework(y_train, framework)
            oracle = oracles.LogisticRegressionOracle(X, y, reg=self.reg)
            return oracle.get_framework(framework=framework,
                                        get_full_batch=get_full_batch)

        def get_outer_oracle(framework="none", get_full_batch=False):
            X = convert_array_framework(X_val, framework)
            y = convert_array_framework(y_val, framework)
            oracle = oracles.LogisticRegressionOracle(X, y)
            return oracle.get_framework(framework=framework,
                                        get_full_batch=get_full_batch)

        def metrics(inner_var, outer_var):
            f_train = get_inner_oracle(framework="none")
            f_val = get_outer_oracle(framework="none")
            inner_star = f_train.inner_var_star(outer_var)
            value_function = f_val.get_value(inner_star, outer_var)
            grad_f_val_inner, grad_f_val_outer = f_val.get_grad(
                inner_star, outer_var
            )
            grad_value = grad_f_val_outer
            v = f_train.get_inverse_hvp(
                inner_star, outer_var,
                grad_f_val_inner
            )
            grad_value -= f_train.get_cross(inner_star, outer_var, v)

            return dict(
                value_func=value_function,
                value=np.linalg.norm(grad_value)**2,
                inner_distance=np.linalg.norm(inner_star-inner_var)**2,
                norm_outer_var=np.linalg.norm(outer_var)**2,
                norm_regul=np.linalg.norm(np.exp(outer_var))**2,
            )

        data = dict(
            get_inner_oracle=get_inner_oracle,
            get_outer_oracle=get_outer_oracle,
            oracle='logreg',
            metrics=metrics,
            n_reg=self.n_reg,
        )
        return data
