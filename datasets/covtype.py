from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils import oracles
    from sklearn.datasets import fetch_covtype
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from benchmark_utils.oracle_utils import convert_array_framework


class Dataset(BaseDataset):

    name = "covtype"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        'oracle': ['multilogreg'],
        'reg': ['exp'],
        'n_reg': ['full'],
        'random_state': [2442],
    }

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        X, y = fetch_covtype(return_X_y=True, download_if_missing=True)
        y -= 1

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=rng
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=.2, random_state=rng
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

        def get_inner_oracle(framework="none", get_full_batch=False):
            X = convert_array_framework(X_train, framework)
            y = convert_array_framework(y_train, framework)
            oracle = oracles.MultiLogRegOracle(X, y,
                                               reg=self.reg)
            return oracle.get_framework(framework=framework,
                                        get_full_batch=get_full_batch)

        def get_outer_oracle(framework="none", get_full_batch=False):
            X = convert_array_framework(X_val, framework)
            y = convert_array_framework(y_val, framework)
            oracle = oracles.MultiLogRegOracle(X, y, reg='none')
            return oracle.get_framework(framework=framework,
                                        get_full_batch=get_full_batch)

        def metrics(inner_var, outer_var):
            f_val = get_outer_oracle(framework="none")
            acc = f_val.accuracy(
                inner_var, outer_var, X_test, y_test
            )
            val_acc = f_val.accuracy(
                inner_var, outer_var, X_val, y_val
            )
            train_acc = f_val.accuracy(
                inner_var, outer_var, X_train, y_train
            )
            return dict(
                train_accuracy=train_acc,
                value=val_acc,
                test_accuracy=acc
            )

        data = dict(
            get_inner_oracle=get_inner_oracle,
            get_outer_oracle=get_outer_oracle,
            oracle='multilogreg',
            metrics=metrics,
            n_reg=self.n_reg,
        )
        return data
