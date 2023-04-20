from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm
    from ..benchmark_utils import oracles


class Dataset(BaseDataset):

    name = "ijcnn1"

    install_cmd = 'conda'
    requirements = ['libsvmdata', 'scikit-learn']
    oracle = 'logreg'
    reg = ['exp', 'lin', 'none']

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

        data = dict(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
        )
        return data, (get_inner_oracle, get_outer_oracle)
