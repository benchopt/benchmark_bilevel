from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    import jax
    import jax.numpy as jnp

    from functools import partial


def loss_sample(inner_var_flat, outer_var, x, y):
    n_classes = y.shape[0]
    n_features = x.shape[0]
    inner_var = inner_var_flat.reshape(n_features, n_classes)
    prod = jnp.dot(x, inner_var)
    lse = jax.nn.logsumexp(prod)
    loss = -jnp.where(y == 1, prod, 0).sum() + lse
    return loss


def loss(theta, lmbda, X, y):
    batched_loss = jax.vmap(loss_sample, in_axes=(None, None, 0, 0))
    return jnp.mean(batched_loss(theta, lmbda, X, y), axis=0)


class Dataset(BaseDataset):

    name = "covtype"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        'reg_parametrization': ['exp'],
        'random_state': [2442],
    }

    def get_data(self):
        assert self.reg_parametrization in ['exp'], (
            f"unknown reg parameter '{self.reg_parametrization}'. "
            "Should be 'lin' or 'exp'."
        )

        rng = np.random.RandomState(self.random_state)
        X, y = fetch_covtype(return_X_y=True, download_if_missing=True)
        y -= 1
        if y.ndim == 1:
            y = OneHotEncoder().fit_transform(y[:, None]).toarray()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=rng
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=.2, random_state=rng
        )
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

            if self.reg_parametrization == 'exp':
                inner_var = inner_var.reshape(self.n_features, self.n_classes)
                alpha = jnp.exp(outer_var)
                res += 0.5 * alpha @ (inner_var * inner_var).sum(axis=0)
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
            return jnp.mean(jnp.argmax(prod, axis=1) != y)

        def metrics(inner_var, outer_var):
            acc = accuracy(inner_var, X_test, y_test)
            val_acc = accuracy(inner_var, X_val, y_val)
            train_acc = accuracy(inner_var, X_train, y_train)
            return dict(
                train_accuracy=float(train_acc),
                value=float(val_acc),
                test_accuracy=float(acc)
            )

        data = dict(
            pb_inner=(f_inner, self.n_samples_inner, self.dim_inner,
                      f_inner_fb),
            pb_outer=(f_outer, self.n_samples_outer, self.dim_outer,
                      f_outer_fb),
            metrics=metrics,
        )
        return data
