from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.datasets import fetch_covtype
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split


class Dataset(BaseDataset):

    name = "covtype"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        'ratio': [0.5, 0.7, 0.9]
    }

    def __init__(self, ratio=.7, random_state=32):
        # Store the parameters of the dataset
        self.random_state = random_state
        self.ratio = ratio

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        ratio = self.ratio
        X, y = fetch_covtype(return_X_y=True, download_if_missing=True)
        y -= 1

        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          test_size=.2,
                                                          random_state=rng)

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                            test_size=.2,
                                                            random_state=rng)
        n_train = y_train.shape[0]
        corrupted = rng.rand(n_train) < ratio
        y_train[corrupted] = rng.randint(0, 7, np.sum(corrupted))
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

        data = dict(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    X_val=X_val, y_val=y_val)
        return data
