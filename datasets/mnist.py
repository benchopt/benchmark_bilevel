from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import pickle
    from urllib import request
    import gzip
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA


class Dataset(BaseDataset):

    name = "mnist"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    def __init__(self, ratio=0.7, random_state=325):
        # Store the parameters of the dataset
        self.random_state = random_state
        self.ratio = ratio

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        ratio = self.ratio
        try:
            with open("mnist.pkl", "rb") as f:
                mnist = pickle.load(f)
        except:
            filename = [
                ["training_images", "train-images-idx3-ubyte.gz"],
                ["test_images", "t10k-images-idx3-ubyte.gz"],
                ["training_labels", "train-labels-idx1-ubyte.gz"],
                ["test_labels", "t10k-labels-idx1-ubyte.gz"],
            ]
            base_url = "http://yann.lecun.com/exdb/mnist/"
            for name in filename:
                print("Downloading " + name[1] + "...")
                request.urlretrieve(base_url + name[1], name[1])
            print("Download complete.")
            mnist = {}
            for name in filename[:2]:
                with gzip.open(name[1], "rb") as f:
                    mnist[name[0]] = np.frombuffer(
                        f.read(), np.uint8, offset=16
                    ).reshape(-1, 28 * 28)
            for name in filename[-2:]:
                with gzip.open(name[1], "rb") as f:
                    mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
            with open("mnist.pkl", "wb") as f:
                pickle.dump(mnist, f)
            print("Save complete.")
            with open("mnist.pkl", "rb") as f:
                mnist = pickle.load(f)

        X_train, y_train, X_test, y_test = (
            mnist["training_images"],
            mnist["training_labels"],
            mnist["test_images"],
            mnist["test_labels"],
        )
        n_train = 2 ** (int(np.floor(np.log2(X_train.shape[0]))))
        n_test = 2 ** (int(np.floor(np.log2(X_test.shape[0]))))

        corrupted = rng.rand(n_train) < ratio
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]
        y_train[corrupted] = rng.randint(0, 10, np.sum(corrupted))
        X_test = X_test[:n_test]
        y_test = y_test[:n_test]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # pca = PCA(30, whiten=True)
        # X_train = pca.fit_transform(X_train)
        # X_test = pca.transform(X_test)
        data = dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        return X_train.shape[1], data
