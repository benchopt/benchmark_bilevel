from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import pickle
    import gzip
    import numpy as np
    from pathlib import Path
    from urllib import request
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split


BASE_URL = "http://yann.lecun.com/exdb/mnist/"
DATA_DIR = Path("./datasets/data")


def download_mnist():
    filenames = {
        "training_images": "train-images-idx3-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "training_labels": "train-labels-idx1-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    for fname in filenames.values():
        print("Downloading " + fname + "...")
        request.urlretrieve(f"{BASE_URL}/{fname}", DATA_DIR / fname)
    print("Download complete.")
    mnist = {}
    for key, fname in filenames.items():
        offset = 16 if "images" in key else 8
        shape = (-1, 28*28) if "images" in key else (-1,)
        with gzip.open(DATA_DIR / fname, "rb") as f:
            mnist[key] = np.frombuffer(
                f.read(), np.uint8, offset=offset
            ).reshape(*shape)
    with open("mnist.pkl", "wb") as f:
        pickle.dump(mnist, f)
    print("Save complete.")


class Dataset(BaseDataset):

    name = "mnist"

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
        if not Path("mnist.pkl").exists():
            download_mnist()

        with open("mnist.pkl", "rb") as f:
            mnist = pickle.load(f)

        X_train, y_train, X_val, y_val = (
            mnist["training_images"],
            mnist["training_labels"],
            mnist["test_images"],
            mnist["test_labels"],
        )
        n_train = 20000
        n_val = 5000
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                            test_size=n_val,
                                                            train_size=n_train,
                                                            random_state=rng)

        corrupted = rng.rand(n_train) < ratio
        y_train[corrupted] = rng.randint(0, 10, np.sum(corrupted))
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)
        data = dict(X_train=X_train, y_train=y_train, X_test=X_test,
                    y_test=y_test, X_val=X_val, y_val=y_val)
        return data
