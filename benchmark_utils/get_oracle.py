def get_oracle(OracleClass, *args, framework=None, **kwargs):
    oracle = OracleClass(*args, **kwargs)
    if framework == "Numba":
        oracle = oracle.numba_oracle
    elif framework == "Jax":
        raise NotImplementedError("Jax oracle not implemented yet")
    elif framework is not None:
        raise ValueError(f"Framework {framework} not supported.")
    return oracle
