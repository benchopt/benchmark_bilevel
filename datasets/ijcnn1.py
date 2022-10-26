from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):

    name = "ijcnn1"

    install_cmd = 'conda'
    requirements = ['libsvmdata', 'scikit-learn']

    def get_data(self):
        X_train, y_train = fetch_libsvm('ijcnn1')
        X_test, y_test = fetch_libsvm('ijcnn1_test')

        data = dict(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )
        return data
