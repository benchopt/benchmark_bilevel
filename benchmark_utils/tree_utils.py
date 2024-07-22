import jax
import jax.numpy as jnp


def update_sgd_fn(var, grad, step_size):
    """
    Helper function that update the variable with a gradient step.

    Parameters
    ----------
    var : pytree
        Variable to update.

    grad : pytree
        Gradient of the variable.

    step_size : float
        Step size of the gradient step.
    """
    return jax.tree_util.tree_map(lambda x, y: x - step_size * y,
                                  var, grad)


def tree_add(a, b):
    """
    Helper function that adds two pytrees.

    Parameters
    ----------
    a : pytree
        First pytree to add.

    b : pytree
        Second pytree to add.
    """
    return jax.tree_util.tree_map(jnp.add, a, b)


def tree_scalar_mult(scalar, tree):
    """
    Helper function that multiplies two pytrees.

    Parameters
    ----------
    a : pytree
        First pytree to multiply.

    b : pytree
        Second pytree to multiply.
    """
    return jax.tree_util.tree_map(lambda x: scalar*x, tree)
