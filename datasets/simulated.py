from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.utils import check_random_state
    from sklearn.model_selection import train_test_split


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [
            (10_000, 50),
        ],
        'sigma_X': [1e-2]
    }

    def __init__(self, n_samples=10, n_features=50, correlation=0,
                 sigma_X=.1, sigma_y=.1, random_state=1):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.correlation = correlation
        self.sigma_X = sigma_X
        self.sigma_y = sigma_y

    def get_data(self):

        rng = check_random_state(self.random_state)
        # Create design matrix with correlated columns
        u = rng.randn(self.n_samples, 1)
        u /= np.linalg.norm(u)
        X = u + self.sigma_X * rng.randn(self.n_samples, self.n_features)

        theta = rng.randn(self.n_features)
        y = X@theta + self.sigma_y * rng.randn(self.n_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=rng
        )

        # print(f"Strong convexity constant: {np.linalg.svd(X_train)[1][-1]}")
        # print(f"Smoothness constant: {np.linalg.norm(X_train, ord=2)**2}")

        data = dict(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )
        return self.n_features, data
