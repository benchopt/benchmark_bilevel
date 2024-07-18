import jax


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
