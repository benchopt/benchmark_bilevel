from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm
    import numpy as np
    from sklearn.preprocessing import normalize


class Dataset(BaseDataset):

    name = "ijcnn1"

    install_cmd = 'conda'
    requirements = ['libsvmdata', 'scikit-learn']

    def get_data(self):
        X_train, y_train = fetch_libsvm('ijcnn1')
        X_test, y_test = fetch_libsvm('ijcnn1_test')
        X_train = X_train.todense()
        X_test = X_test.todense()
        n_train = 2 ** (int(np.floor(np.log2(X_train.shape[0]))) - 3)
        n_test = 2 ** (int(np.floor(np.log2(X_test.shape[0]))))
        X_train = X_train[:n_train]
        # X_train = normalize(X_train)
        # X_test = normalize(X_test)
        y_train = y_train[:n_train]
        X_test = X_test[:n_test]
        y_test = y_test[:n_test]
        data = dict(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )
        return X_train.shape[1], data
