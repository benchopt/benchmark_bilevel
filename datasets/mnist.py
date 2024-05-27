from pathlib import Path

from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import pickle
    import gzip
    import numpy as np
    from urllib import request
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    import jax
    import jax.numpy as jnp
    from jax.nn import logsumexp, sigmoid

    from functools import partial


BASE_URL = "http://yann.lecun.com/exdb/mnist/"
DATA_DIR = Path(__file__).parent / "data"


def download_mnist():
    # Make sure the data exists
    DATA_DIR.mkdir(exist_ok=True)

    # download the archives.
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


def loss_sample(inner_var_flat, outer_var, x, y):
    n_classes = y.shape[0]
    n_features = x.shape[0]
    inner_var = inner_var_flat.reshape(n_features, n_classes)
    prod = jnp.dot(x, inner_var)
    lse = logsumexp(prod)
    loss = -jnp.where(y == 1, prod, 0).sum() + lse
    return loss


def loss(inner_var, outer_var, X, y):
    batched_loss = jax.vmap(loss_sample, in_axes=(None, None, 0, 0))
    return jnp.mean(batched_loss(inner_var, outer_var, X, y), axis=0)


def weighted_loss(inner_var, outer_var, X, y):
    weights = sigmoid(outer_var)
    batched_loss = jax.vmap(loss_sample, in_axes=(None, None, 0, 0))
    return jnp.mean(weights * batched_loss(inner_var, outer_var, X, y),
                    axis=0)


class Dataset(BaseDataset):

    name = "mnist"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        'ratio': [0.5, 0.7, 0.9],
        'random_state': [32],
        'oracle': ['datacleaning'],
        'reg': [2e-1]
    }

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        ratio = self.ratio
        if not Path("mnist.pkl").exists():
            download_mnist()

        with open("mnist.pkl", "rb") as f:
            mnist = pickle.load(f)

        X_train, y_train, X_test, y_test = (
            mnist["training_images"],
            mnist["training_labels"],
            mnist["test_images"],
            mnist["test_labels"],
        )
        n_train = 20000
        n_val = 5000
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=n_val,
                                                          train_size=n_train,
                                                          random_state=rng)

        corrupted = rng.rand(n_train) < ratio
        y_train[corrupted] = rng.randint(0, 10, np.sum(corrupted))
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)
        X_val = jnp.array(X_val)
        y_val = jnp.array(y_val)
        X_test = jnp.array(X_test)
        y_test = jnp.array(y_test)

        if y_train.ndim == 1:
            y_train = OneHotEncoder().fit_transform(y_train[:, None]).toarray()
        if y_val.ndim == 1:
            y_val = OneHotEncoder().fit_transform(y_val[:, None]).toarray()
        if y_test.ndim == 1:
            y_test = OneHotEncoder().fit_transform(y_test[:, None]).toarray()

        self.n_features = X_train.shape[1]
        self.n_classes = y_train.shape[1]

        self.n_samples_inner = X_train.shape[0]
        self.n_samples_outer = X_val.shape[0]

        self.dim_inner = self.n_features * self.n_classes
        self.dim_outer = self.n_classes

        @partial(jax.jit, static_argnames=('batch_size'))
        def f_inner(inner_var, outer_var, start=0, batch_size=1):
            x = jax.lax.dynamic_slice(
                X_train, (start, 0), (batch_size, X_train.shape[1])
            )
            y = jax.lax.dynamic_slice(
                y_train, (start, 0), (batch_size, self.n_classes)
            )
            res = loss(inner_var, outer_var, x, y)

            res += self.reg * np.sum(inner_var**2)
            return res

        @partial(jax.jit, static_argnames=('batch_size'))
        def f_outer(inner_var, outer_var, start=0, batch_size=1):
            x = jax.lax.dynamic_slice(
                X_val, (start, 0), (batch_size, X_val.shape[1])
            )
            y = jax.lax.dynamic_slice(
                y_val, (start, 0), (batch_size, self.n_classes)
            )
            res = loss(inner_var, outer_var, x, y)
            return res

        f_inner_fb = partial(f_inner, start=0,
                             batch_size=self.n_samples_inner)
        f_outer_fb = partial(f_outer, start=0,
                             batch_size=self.n_samples_outer)

        @jax.jit
        def accuracy(inner_var, X, y):
            if y.ndim == 2:
                y = y.argmax(axis=1)
            inner_var = inner_var.reshape(self.n_features,
                                          self.n_classes)
            prod = X @ inner_var
            return np.mean(np.argmax(prod, axis=1) != y)

        def metrics(inner_var, outer_var):
            acc = accuracy(inner_var, X_test, y_test)
            val_acc = accuracy(inner_var, X_val, y_val)
            train_acc = accuracy(inner_var, X_train, y_train)
            return dict(
                train_accuracy=train_acc,
                value=val_acc,
                test_accuracy=acc
            )

        data = dict(
            pb_inner=(f_inner, self.n_samples_inner, self.dim_inner,
                      f_inner_fb),
            pb_outer=(f_outer, self.n_samples_outer, self.dim_outer,
                      f_outer_fb),
            metrics=metrics,
            oracle='datacleaning',
            n_reg=None
        )
        return data
