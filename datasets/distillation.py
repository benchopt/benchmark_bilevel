from pathlib import Path

from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import pickle
    import gzip
    import numpy as np
    from urllib import request
    from sklearn.preprocessing import StandardScaler

    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import optax
    from functools import partial

    from benchmark_utils.tree_utils import tree_inner_product


BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"
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


class Dataset(BaseDataset):
    """Distillation on mnist"""

    name = "distillation"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        'n_distilled_images': [10],
        'random_state': [32],
        'reg': [1e-1]
    }

    def get_data(self):
        n_samples_inner = self.n_distilled_images
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
        self.n_features = X_train.shape[1]
        self.n_classes = 10

        if n_samples_inner < self.n_classes:
            raise ValueError("n_distilled_images must be >= n_classes")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)
        X_test = jnp.array(X_test)
        y_test = jnp.array(y_test)

        self.n_samples_outer = X_train.shape[0]

        self.dim_outer = n_samples_inner

        class CNN(nn.Module):
            """A simple CNN model."""
            @nn.compact
            def __call__(self, x):
                x = nn.Conv(features=32, kernel_size=(3, 3))(x)
                x = nn.relu(x)
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                x = nn.Conv(features=64, kernel_size=(3, 3))(x)
                x = nn.relu(x)
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                x = x.reshape((x.shape[0], -1))  # flatten
                x = nn.Dense(features=256)(x)
                x = nn.relu(x)
                x = nn.Dense(features=10)(x)
                return x

        cnn = CNN()
        jax_rng = jax.random.PRNGKey(self.random_state)
        inner_params = cnn.init(jax_rng, jnp.ones([1, 28, 28, 1]))['params']
        self.dim_inner = jax.tree_map(lambda x: x.shape, inner_params)
        self.dim_outer = (n_samples_inner, 28, 28, 1)
        if self.n_distilled_images == self.n_classes:
            outer_labels = jnp.arange(self.n_classes)
        else:
            outer_labels = jax.random.randint(jax_rng, (n_samples_inner,),
                                              0, 10)
            # ensure that all classes are present
            while len(jnp.unique(outer_labels)) < self.n_classes:
                jax_rng = jax.random.split(jax_rng)[0]
                outer_labels = jax.random.randint(jax_rng, (n_samples_inner,),
                                                  0, 10)

        def loss(params, x, y):
            logits = cnn.apply({'params': params}, x)
            one_hot = jax.nn.one_hot(y, 10)
            return jnp.mean(optax.softmax_cross_entropy(logits=logits,
                                                        labels=one_hot))

        @partial(jax.jit, static_argnames=('batch_size'))
        def f_inner(inner_var, outer_var, start=0, batch_size=1):

            # outer_labels_batch = jax.lax.dynamic_slice(
            #     outer_labels, (start,), (batch_size,)
            # )
            # outer_var_batch = jax.lax.dynamic_slice(
            #     outer_var, (start, 0, 0, 0), (batch_size, 28, 28, 1)
            # )

            # I think since the distilled dataset is supposed to be very small
            # (typically one sample per class), we can just use the whole
            # distilled dataset
            reg = self.reg * tree_inner_product(inner_var, inner_var)
            return loss(inner_var, outer_var, outer_labels) + reg

        @partial(jax.jit, static_argnames=('batch_size'))
        def f_outer(inner_var, outer_var, start=0, batch_size=1):
            x = jax.lax.dynamic_slice(
                X_train, (start, 0), (batch_size, X_train.shape[1])
            ).reshape((batch_size, 28, 28, 1))
            y = jax.lax.dynamic_slice(
                y_train, (start,), (batch_size,)
            )
            res = loss(inner_var, x, y)
            return res

        f_inner_fb = partial(f_inner, start=0,
                             batch_size=self.n_distilled_images)
        f_outer_fb = partial(f_outer, start=0,
                             batch_size=self.n_samples_outer)

        @jax.jit
        def accuracy(inner_var, X, y):
            if y.ndim == 2:
                y = y.argmax(axis=1)
            logits = cnn.apply({'params': inner_var},
                               X.reshape((-1, 28, 28, 1)))
            return jnp.mean(jnp.argmax(logits, axis=1) == y)

        def metrics(inner_var, outer_var):
            acc = accuracy(inner_var, X_test, y_test)
            train_acc = accuracy(inner_var, X_train, y_train)
            distilled_acc = accuracy(inner_var, outer_var, outer_labels)
            return dict(
                train_accuracy=float(train_acc),
                test_accuracy=float(acc),
                distilled_accuracy=float(distilled_acc),
                # sanity check, should be 1.0
            )

        data = dict(
            pb_inner=(f_inner, self.n_distilled_images, self.dim_inner,
                      f_inner_fb),
            pb_outer=(f_outer, self.n_samples_outer, self.dim_outer,
                      f_outer_fb),
            metrics=metrics,
        )
        return data
