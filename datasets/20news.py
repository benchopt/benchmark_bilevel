
from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.utils import check_random_state
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_20newsgroups_vectorized
    scipy_to_csrmatrix = import_ctx.import_from(
        'sparse_matrix', 'scipy_to_csrmatrix'
    )


class Dataset(BaseDataset):

    name = "20news"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def __init__(self, random_state=27):
        self.random_state = random_state

    def get_data(self):
        rng = check_random_state(self.random_state)
        X, y = fetch_20newsgroups_vectorized(subset='train', return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=rng
        )

        data = dict(
            X_train=scipy_to_csrmatrix(X_train), y_train=y_train,
            X_test=scipy_to_csrmatrix(X_test), y_test=y_test
        )
        return X.shape[1], data
