from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import jax
    import numpy as np
    import jax.numpy as jnp
    from functools import partial
    from benchmark_utils.gen_matrices import gen_matrices


def get_hessian_min_eigval(hess_inner_inner, cross_inner, hess_outer_inner,
                           hess_outer_outer, cross_outer):
    """Compute the minimum eigenvalue of the Hessian of the value function."""

    jac_z_star = - np.linalg.solve(hess_inner_inner, cross_inner.T)
    hess = jac_z_star.T.dot(hess_outer_inner) @ jac_z_star + hess_outer_outer

    tmp = cross_outer @ jac_z_star
    hess += .5 * (tmp + tmp.T)
    return np.min(np.linalg.eigvalsh(hess))


def quadratic(inner_var, outer_var, hess_inner, hess_outer, cross,
              linear_inner, linear_outer):
    res = .5 * inner_var @ (hess_inner @ inner_var)
    res += .5 * outer_var @ (hess_outer @ outer_var)
    res += outer_var @ cross @ inner_var
    res += linear_inner @ inner_var
    res += linear_outer @ outer_var
    return res


def batched_quadratic(inner_var, outer_var, hess_inner, hess_outer, cross,
                      linear_inner, linear_outer):
    batched_loss = jax.vmap(quadratic, in_axes=(None, None, 0, 0, 0, 0, 0))
    return jnp.mean(
        batched_loss(inner_var, outer_var, hess_inner, hess_outer,
                     cross, linear_inner, linear_outer)
    )


def get_function(hess_inner, hess_outer, cross, linear_inner, linear_outer):

    @partial(jax.jit, static_argnames=('batch_size'))
    def f(inner_var, outer_var, start=0, batch_size=1):
        hess_inner_batch = jax.lax.dynamic_slice(
            hess_inner, (start, 0, 0), (batch_size, *hess_inner.shape[1:])
        )
        hess_outer_batch = jax.lax.dynamic_slice(
            hess_outer, (start, 0, 0), (batch_size, *hess_outer.shape[1:])
        )
        cross_mat_batch = jax.lax.dynamic_slice(
            cross, (start, 0, 0), (batch_size, *cross.shape[1:])
        )
        linear_inner_batch = jax.lax.dynamic_slice(
            linear_inner, (start, 0),
            (batch_size, linear_inner.shape[1])
        )
        linear_outer_batch = jax.lax.dynamic_slice(
            linear_outer, (start, 0),
            (batch_size, linear_outer.shape[1])
        )
        return batched_quadratic(
            inner_var, outer_var, hess_inner_batch, hess_outer_batch,
            cross_mat_batch, linear_inner_batch, linear_outer_batch
        )
    return f


class Dataset(BaseDataset):

    name = "simulated"

    parameters = {
        'oracle': ['quadratic'],
        'L_inner_inner': [1.],
        "L_inner_outer": [1.],
        'mu_inner': [.1],
        'L_outer_inner': [1.],
        "L_outer_outer": [1.],
        'L_cross_inner': [.1],
        "L_cross_outer": [.1],
        'random_state': [2442],
        'n_samples_inner': [1024],
        'n_samples_outer': [1024],
        'dim_inner': [100],
        'dim_outer': [100],
    }

    def get_data(self):
        key = jax.random.PRNGKey(self.random_state)

        for k in range(20):
            keys = jax.random.split(key, 2)

            (hess_inner_inner, hess_inner_outer, cross_inner,
             linear_inner_inner, linear_inner_outer) = gen_matrices(
                self.n_samples_inner,
                self.dim_inner,
                self.dim_outer,
                self.L_inner_inner,
                self.L_outer_inner,
                self.L_cross_inner,
                self.mu_inner,
                keys[0],
            )

            (hess_outer_inner, hess_outer_outer, cross_outer,
             linear_outer_inner, linear_outer_outer) = gen_matrices(
                self.n_samples_outer,
                self.dim_inner,
                self.dim_outer,
                self.L_inner_outer,
                self.L_outer_outer,
                self.L_cross_outer,
                self.mu_inner,
                keys[1]
             )

            hess_inner_inner_fb = jnp.mean(hess_inner_inner, axis=0)
            hess_outer_inner_fb = jnp.mean(hess_outer_inner, axis=0)
            cross_inner_fb = jnp.mean(cross_inner, axis=0)

            hess_inner_outer_fb = jnp.mean(hess_inner_outer, axis=0)
            linear_inner_outer_fb = jnp.mean(linear_inner_outer, axis=0)

            hess_outer_outer_fb = jnp.mean(hess_outer_outer, axis=0)
            cross_outer_fb = jnp.mean(cross_outer, axis=0)

            eig = get_hessian_min_eigval(
                hess_inner_inner_fb, cross_inner_fb, hess_outer_inner_fb,
                hess_outer_outer_fb, cross_outer_fb
            )

            if eig >= 1e-12:
                break
        else:
            raise ValueError("Could not generate a dataset with a "
                             "positive Hessian.")
        print(
            f"Generated dataset with a positive Hessian after {k+1} trial(s)."
        )
        print(f"Minimum eigenvalue of the Hessian: {eig}")
        linear_inner_inner_fb = jnp.mean(linear_inner_inner, axis=0)

        linear_outer_inner_fb = jnp.mean(linear_outer_inner, axis=0)
        linear_outer_outer_fb = jnp.mean(linear_outer_outer, axis=0)
        f_inner = get_function(
            hess_inner_inner, hess_inner_outer, cross_inner,
            linear_inner_inner, linear_inner_outer
        )

        f_inner_fb = get_function(
            hess_inner_inner_fb[None], hess_inner_outer_fb[None],
            cross_inner_fb[None], linear_inner_inner_fb[None],
            linear_inner_outer_fb[None]
        )

        f_outer = get_function(
            hess_outer_inner, hess_outer_outer, cross_outer,
            linear_outer_inner, linear_outer_outer
        )

        f_outer_fb = get_function(
            hess_outer_inner_fb[None], hess_outer_outer_fb[None],
            cross_outer_fb[None], linear_outer_inner_fb[None],
            linear_outer_outer_fb[None]
        )

        grad_outer = jax.jit(jax.grad(f_outer_fb, argnums=(0, 1)))

        def metrics(inner_var, outer_var):
            inner_sol = jnp.linalg.solve(
                hess_inner_inner_fb,
                - linear_inner_inner_fb - cross_inner_fb.T @ outer_var
            )
            grad_in, grad_out = grad_outer(inner_sol, outer_var)
            v_sol = - jnp.linalg.solve(
                hess_inner_inner_fb,
                grad_in
            )
            grad_value = grad_out
            grad_value += cross_inner_fb @ v_sol
            return dict(
                func=float(f_outer(inner_sol, outer_var)),
                value=np.linalg.norm(grad_value)**2,
                inner_distance=np.linalg.norm(inner_sol - inner_var)**2,
            )

        data = dict(
            pb_inner=(f_inner, self.n_samples_inner, self.dim_inner,
                      f_inner_fb),
            pb_outer=(f_outer, self.n_samples_outer, self.dim_outer,
                      f_outer_fb),
            metrics=metrics,
            n_reg=None,
        )
        return data
