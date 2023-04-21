from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from libsvmdata import fetch_libsvm
    from benchmark_utils import oracles


class Dataset(BaseDataset):

    name = "ijcnn1"

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata', 'pip:scikit-learn']

    parameters = {
        'oracle': ['logreg'],
        'reg': ['exp'],
        'n_reg': ['full'],
    }

    def get_data(self):
        X_train, y_train = fetch_libsvm('ijcnn1')
        X_val, y_val = fetch_libsvm('ijcnn1_test')

        def get_inner_oracle(framework=None):
            if self.oracle == 'logreg':
                oracle = oracles.LogisticRegressionOracle(
                    X_train, y_train, reg=self.reg
                )
                if framework == "Numba":
                    oracle = oracle.numba_oracle
                elif framework == "Jax":
                    raise NotImplementedError("Jax oracle not implemented yet")
                elif framework is not None:
                    raise ValueError(f"Framework {framework} not supported.")
            else:
                raise ValueError(f"Oracle {self.oracle} not supported.")
            return oracle

        def get_outer_oracle(framework=None):
            if self.oracle == 'logreg':
                oracle = oracles.LogisticRegressionOracle(
                    X_val, y_val
                )
                if framework == "Numba":
                    oracle = oracle.numba_oracle
                elif framework == "Jax":
                    raise NotImplementedError("Jax oracle not implemented yet")
                elif framework is not None:
                    raise ValueError(f"Framework {framework} not supported.")
            else:
                raise ValueError(f"Oracle {self.oracle} not supported.")
            return oracle

        def metrics(inner_var, outer_var):
            f_train = get_inner_oracle(framework=None)
            f_val = get_outer_oracle(framework=None)
            inner_star = f_train.get_inner_var_star(outer_var)
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
            )

        data = dict(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            get_inner_oracle=get_inner_oracle,
            get_outer_oracle=get_outer_oracle,
            oracle=self.oracle,
            metrics=metrics,
            n_reg=self.n_reg,
        )
        return data
