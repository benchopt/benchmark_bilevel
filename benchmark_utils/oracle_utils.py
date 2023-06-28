import jax.numpy as jnp


def convert_array_framework(x, framework=None):
    if framework == "jax":
        x = jnp.array(x)
    return x
