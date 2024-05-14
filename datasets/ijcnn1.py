from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from libsvmdata import fetch_libsvm

    import jax
    import jax.numpy as jnp
    from functools import partial
    from jax.nn import log_sigmoid

    from jaxopt import LBFGS


def loss_sample(inner_var, outer_var, x, y):
    return -log_sigmoid(y*jnp.dot(inner_var, x))


@jax.jit
def loss(inner_var, outer_var, X, y):
    batched_loss = jax.vmap(loss_sample, in_axes=(None, None, 0, 0))
    return jnp.mean(batched_loss(inner_var, outer_var, X, y), axis=0)


class Dataset(BaseDataset):

    name = "ijcnn1"

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata', 'scikit-learn']

    parameters = {
        'reg': ['exp'],
        'n_reg': ['full'],
        'oracle': ['logreg'],
    }

    def get_data(self):
        X_train, y_train = fetch_libsvm('ijcnn1')
        X_val, y_val = fetch_libsvm('ijcnn1_test')

        X_train, y_train = jnp.array(X_train), jnp.array(y_train)
        X_val, y_val = jnp.array(X_val), jnp.array(y_val)

        self.n_samples_inner = X_train.shape[0]
        self.dim_inner = X_train.shape[1]
        self.n_samples_outer = X_val.shape[0]
        self.dim_outer = X_val.shape[1]

        @partial(jax.jit, static_argnames=('batch_size'))
        def f_inner(inner_var, outer_var, start=0, batch_size=1):
            x = jax.lax.dynamic_slice(
                X_train, (start, 0), (batch_size, X_train.shape[1])
            )
            y = jax.lax.dynamic_slice(
                y_train, (start, ), (batch_size, )
            )
            res = loss(inner_var, outer_var, x, y)

            if self.reg == 'exp':
                res += jnp.dot(jnp.exp(outer_var) * inner_var, inner_var)/2
            elif self.reg == 'lin':
                res += jnp.dot(outer_var * inner_var, inner_var)/2
            return res

        @partial(jax.jit, static_argnames=('batch_size'))
        def f_outer(inner_var, outer_var, start=0, batch_size=1):
            x = jax.lax.dynamic_slice(
                X_val, (start, 0), (batch_size, X_val.shape[1])
            )
            y = jax.lax.dynamic_slice(
                y_val, (start, ), (batch_size, )
            )
            res = loss(inner_var, outer_var, x, y)
            return res

        f_inner_fb = partial(
            f_inner, batch_size=X_train.shape[0], start=0
        )
        f_outer_fb = partial(
            f_outer, batch_size=X_val.shape[0], start=0
        )

        solver_inner = LBFGS(fun=f_inner_fb)

        def value_function(outer_var):
            inner_var_star = solver_inner.run(
                jnp.zeros(X_train.shape[1]), outer_var
            ).params

            return f_outer_fb(inner_var_star, outer_var), inner_var_star

        value_and_grad = jax.jit(
            jax.value_and_grad(value_function, has_aux=True)
        )

        def metrics(inner_var, outer_var):
            (value_fun, inner_star), grad_value = value_and_grad(outer_var)
            return dict(
                value_func=float(value_fun),
                value=float(jnp.linalg.norm(grad_value)**2),
                inner_distance=float(jnp.linalg.norm(inner_star-inner_var)**2),
                norm_outer_var=float(jnp.linalg.norm(outer_var)**2),
                norm_regul=float(jnp.linalg.norm(np.exp(outer_var))**2),
            )

        data = dict(
            pb_inner=(f_inner, self.n_samples_inner, self.dim_inner),
            pb_outer=(f_outer, self.n_samples_outer, self.dim_outer),
            metrics=metrics,
            n_reg=None,
            oracle='logreg',
            reg=self.reg
        )
        return data
