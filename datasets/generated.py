from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.model_selection import train_test_split


class Dataset(BaseDataset):

    name = "Generated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [
            # (50_000, 22)
            (10_000, 20)
        ]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=325):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        beta = rng.randn(self.n_features)
        X = rng.randn(self.n_samples, self.n_features)
        X /= X.std(axis=0, keepdims=True)
        noise = 0.1 * rng.randn(self.n_samples)
        indiv_noises = 10 * rng.rand(self.n_features)
        indiv_noises[:self.n_features // 2] /= 10
        beta_noise = indiv_noises * rng.rand(self.n_samples, self.n_features)
        y = np.sum(beta * (1 + beta_noise) * X, axis=1) + noise

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=rng
        )

        data = dict(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )

        return self.n_features, data
