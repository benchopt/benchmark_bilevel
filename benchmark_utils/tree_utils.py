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


def tree_inner_product(a, b):
    """
    Helper function that computes the inner product of two pytrees.

    Parameters
    ----------
    a : pytree
        First pytree.

    b : pytree
        Second pytree.
    """
    return jax.tree_util.tree_reduce(jnp.add, jax.tree_util.tree_map(
        lambda x, y: jnp.sum(x * y), a, b))


def init_memory_of_trees(n_memories, tree):
    """
    Helper function that initializes the memory of a pytree.

    Parameters
    ----------
    n_memories : int
        Number of memories to initialize.

    tree : pytree
        Pytree to initialize.
    """
    return jax.tree_util.tree_map(lambda x: jnp.zeros((n_memories, *x.shape)),
                                  tree)


def select_memory(memory, idx):
    """
    Helper function that selects a memory from a memory pytree.

    Parameters
    ----------
    memory : pytree
        Memory pytree.

    idx : int
        Index of the memory to select.
    """
    return jax.tree_util.tree_map(lambda x: x[idx], memory)


def update_memory(memory, idx, value):
    """
    Helper function that updates a memory from a memory pytree.

    Parameters
    ----------
    memory : pytree
        Memory pytree.

    idx : int
        Index of the memory to update.

    value : pytree
        Value to update the memory with.
    """
    return jax.tree_util.tree_map(lambda x: x.at[idx].set(value), memory)
