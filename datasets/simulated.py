from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.datasets import make_correlated_data
    from sklearn.model_selection import train_test_split


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [
            (1000, 10),
        ],
        'correlation': [0, .5, 1-10**(-10), 1-10**(-15)]

    }

    def __init__(self, n_samples=10, n_features=50, correlation=0,
                 sigma=.1, random_state=None):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.correlation = correlation
        self.sigma = sigma

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        # Create design matrix with correlated columns
        X_T, _, _ = make_correlated_data(
            n_samples=self.n_features,
            n_features=self.n_samples,
            rho=self.correlation,
        )
        X = X_T.T
        theta = rng.randn(self.n_features)
        y = X@theta + self.sigma * rng.randn(self.n_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=rng
        )

        data = dict(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )
        return self.n_features, data
