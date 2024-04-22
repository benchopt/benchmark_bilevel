import jax
import jax.numpy as jnp
from joblib import Memory

memory = Memory("__cache__",
                verbose=False)


@memory.cache
def gen_matrices(n_samples, d_inner, d_outer, L_inner, L_outer,
                 L_cross, mu, key=jax.random.PRNGKey(0)):
    keys = jax.random.split(key, 10)

    # Generate H^1/2
    U, *_ = jnp.linalg.svd(jax.random.normal(keys[0], (d_inner, 1)))
    D = jnp.logspace(jnp.log10(mu) / 2, jnp.log10(L_inner)/2, d_inner)
    D = jnp.diag(D)
    A = U @ D @ U.T

    # Generate x with correlation matrix H (spectrum [mu, 1])
    # and take H_i as empirical correlation x.x^T
    X = jax.random.normal(keys[1], (n_samples, 1, d_inner)) @ A
    hess_inner = X.transpose(0, 2, 1) @ X / X.shape[1]

    # Generate H^1/2
    U, *_ = jnp.linalg.svd(jax.random.normal(keys[2], (d_outer, 1)))
    D = jnp.logspace(jnp.log10(mu)/2, jnp.log10(L_outer)/2, d_outer-1)
    D = jnp.diag(jnp.r_[0, D])
    A = U @ D @ U.T

    # Generate x with correlation matrix H (spectrum [mu, 1])
    # and take H_i as empirical correlation x.x^T
    X = jax.random.normal(keys[3], (n_samples, 1, d_outer)) @ A
    hess_outer = X.transpose(0, 2, 1) @ X / X.shape[1]

    # Generate singular vectors and values for rectangular correlation matrix
    U, _, V = jnp.linalg.svd(jax.random.normal(keys[4], (d_outer, d_inner)))
    D = jnp.zeros((d_outer, d_inner))
    D[:min(d_outer, d_inner), :min(d_outer, d_inner)] = jnp.diag(
        jnp.logspace(jnp.log10(mu), jnp.log10(L_cross), min(d_outer, d_inner)))

    # Generate x with correlation matrix UDV^T
    X = jax.random.normal(keys[5], (n_samples, 1, d_inner))
    X1 = U @ D @ X.transpose(0, 2, 1)
    X2 = X @ V.T

    cross_mat = X1 @ X2

    return (
        hess_inner, hess_outer,
        cross_mat,
        jax.random.normal(keys[6], (n_samples, d_inner)),
        jax.random.normal(keys[7], (n_samples, d_outer))
    )