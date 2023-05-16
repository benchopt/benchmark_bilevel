import jax.numpy as jnp


def get_oracle(OracleClass, *args, framework=None, get_fb=False,
               **kwargs):
    oracle = OracleClass(*args, **kwargs)
    if framework == "numba":
        oracle = oracle.numba_oracle
    elif framework == "jax":
        if get_fb:
            oracle = oracle.jax_oracle, oracle.jax_oracle_fb
        else:
            oracle = oracle.jax_oracle
    elif framework is not None:
        raise ValueError(f"Framework {framework} not supported.")
    return oracle


def convert_array_framework(x, framework=None):
    if framework == "jax":
        x = jnp.array(x)
    return x
