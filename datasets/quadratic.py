from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.utils import check_random_state


class Dataset(BaseDataset):

    name = "Quadratic"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_z, n_x': [
            (2, 2),
        ],
    }

    def __init__(self, n_z, n_x, random_state=1):
        # Store the parameters of the dataset
        self.n_z = n_z
        self.n_x = n_x
        self.random_state = random_state

    def get_data(self):

        rng = check_random_state(self.random_state)
        # Create design matrix with correlated columns
        A_z_inner = rng.randn(self.n_z, self.n_z)
        U, _, V = np.linalg.svd(A_z_inner)
        A_z_inner = U.dot(np.diag(np.logspace(-1, 1, self.n_z))).dot(U.T)

        A_x_inner = rng.randn(self.n_x, self.n_x)
        U, _, V = np.linalg.svd(A_x_inner)
        A_x_inner = U.dot(np.diag(np.logspace(-1, 1, self.n_z))).dot(U.T)

        b_inner = rng.randn()
        c_inner = rng.randn()

        A_z_outer = rng.randn(self.n_z, self.n_z)
        A_z_outer = A_z_outer.dot(A_z_outer)

        A_x_outer = rng.randn(self.n_x, self.n_x)
        A_x_outer = A_x_outer.dot(A_x_outer)

        b_outer = rng.randn()
        c_outer = rng.randn()

        data = dict(
            A_z_inner=A_z_inner, A_x_inner=A_x_inner, b_inner=b_inner,
            c_inner=c_inner,
            A_z_outer=A_z_outer, A_x_outer=A_x_outer, b_outer=b_outer,
            c_outer=c_outer
        )
        return self.n_z, data
