import numpy as np
from sklearn.model_selection import train_test_split

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [
            (1000, 500),
            (5000, 200)]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=rng
        )

        data = dict(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )

        return self.n_features, data
