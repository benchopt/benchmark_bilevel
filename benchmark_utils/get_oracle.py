def get_oracle(OracleClass, *args, framework=None, **kwargs):
    oracle = OracleClass(*args, **kwargs)
    if framework == "numba":
        oracle = oracle.numba_oracle
    elif framework == "jax":
        raise NotImplementedError("Jax oracle not implemented yet")
    elif framework is not None:
        raise ValueError(f"Framework {framework} not supported.")
    return oracle
