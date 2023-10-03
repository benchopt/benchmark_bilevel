from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils import check_random_state


class BaseOracle(ABC):
    """A base class to compute all the oracles of a function needed in order
    to run stochastic first order bilevel optimization procedures.

    Oracle should implement:

    - `value(inner_var, outer_var, idx)`: should return the value of the
      function at the points inner_var and outer_var estimated on the indices
      contained in idx.

    - `grad_inner_var(inner_var, outer_var, idx): should return the gradient
      with respect to the inner variable inner_var at inner_var and outer_var
      estimated on the indices contained in idx.

    - `grad_outer_var(inner_var, outer_var, idx): should return the gradient
      with respect to the outer variable outer_var at inner_var and x estimated
      on the indices contained in idx.

    - `cross(inner_var, outer_var, v, idx): should return the product between
      the matrix of the cross derivatives and a vector v. If M is the cross
      derivatives matrix, M[i, j] is the derivative of the function with
      respect to outer_var_i and inner_var_j at inner_var and outer_var
      estimated on the indices contained in idx.

    - `hvp(inner_var, outer_var, v, idx): should return the product between
      the Hessian (with respect to the inner variable) at inner_var and
      outer_var estimated on the indices contained in idx and a vector v.

    - `inverse_hvp(inner_var, outer_var, v, idx): should
      return the product between the inverse Hessian (with respect to the inner
      variable) at inner_var and outer_var estimated on the indices contained
      in idx and a vector v.

    Note that the batch size should be defined in __init__.
    """
    # Shape of the variable for the considered problem
    variables_shape = None

    def __init__(self):
        self.memory = {}

    @abstractmethod
    def value(self, inner_var, outer_var, idx):
        pass

    @abstractmethod
    def grad_inner_var(self, inner_var, outer_var, idx):
        pass

    @abstractmethod
    def grad_outer_var(self, inner_var, outer_var, idx):
        pass

    @abstractmethod
    def cross(self, inner_var, outer_var, v, idx):
        pass

    @abstractmethod
    def hvp(self, inner_var, outer_var, v, idx):
        pass

    @abstractmethod
    def inverse_hvp(self, inner_var, outer_var, v, idx, approx='cg'):
        pass

    def grad(self, inner_var, outer_var, idx):
        return self.grad_inner_var(inner_var, outer_var, idx), \
            self.grad_outer_var(inner_var, outer_var, idx)

    def prox(self, inner_var, outer_var):
        "Prox function for the inner and outer_var."
        return inner_var, outer_var

    def oracles(self, inner_var, outer_var, v, idx, inverse='id'):
        """Compute all the quantities together on the same batch."""
        val = self.value(inner_var, outer_var, idx)
        grad = self.grad_inner_var(inner_var, outer_var, idx)
        hvp = self.hvp(inner_var, outer_var, v, idx)
        inv_hvp = self.inverse_hvp(
            inner_var, outer_var, v, idx, approx=inverse
        )
        implicit_grad = self.cross(inner_var, outer_var, inv_hvp, idx)
        return val, grad, hvp, implicit_grad

    def inner_var_star(self, outer_var):
        inner_shape, _ = self.variables_shape
        var_shape_flat = np.prod(inner_shape)

        def func(inner_var):
            inner_var = inner_var.reshape(*inner_shape)
            return self.get_value(inner_var, outer_var)

        def fprime(inner_var):
            inner_var = inner_var.reshape(*inner_shape)
            return self.get_grad_inner_var(inner_var, outer_var)

        inner_var_star, _, d = fmin_l_bfgs_b(
            func, np.zeros(var_shape_flat), fprime=fprime, maxls=30
        )

        if d['warnflag'] != 0:
            print('LBFGS did not converged!')
            print("Final gradient:", d['grad'])
            raise RuntimeError()
        return inner_var_star

    @abstractmethod
    def _get_numba_oracle(self):
        pass

    @abstractmethod
    def _get_jax_oracle(self, get_full_batch=False):
        pass

    def get_framework(self, framework='none', get_full_batch=False):
        """
        Returns the oracle in the desired framework.

        Parameters
        ----------
        framework : str, default='none'
            The framework in which the oracle should be returned. Should be in
            ['none', 'jax', 'numba'].
        get_full_batch : bool, default=False
            If False, returns the oracle with the batch size defined in
            __init__.
            If True, returns a tuple (oracle, oracle_fb) where oracle is the
            oracle with the batch size defined in __init__ and oracle_fb is the
            oracle with a batch size equal to the number of samples. It is
            useful only for the jax framework.

        Returns
        -------
        oracle : Oracle class of callable
            The oracle in the desired framework. If framewors is 'none' or
            'numba', returns an Oracle class. If framework is 'jax', returns a
            differentiable function.
        """
        if framework == 'none':
            return self
        elif framework == 'jax':
            return self._get_jax_oracle(get_full_batch=get_full_batch)
        elif framework == 'numba':
            return self._get_numba_oracle()

    def __getattr__(self, name):
        # construct get_* and get_batch_* for all methods in this list:

        if name.startswith('get_batch_'):
            name = name.replace('get_batch_', '')
            method = getattr(self, name)
            return _get_batch_method(method, name).__get__(self, BaseOracle)

        if name.startswith('get_'):
            method = getattr(self, name.replace('get_', ''))

            return _get_full_batch_method(method).__get__(self, BaseOracle)

        return super().__getattribute__(name)


def _get_full_batch_method(method):

    def get_full_batch(self, *args, **kwargs):
        idx = np.arange(self.n_samples)
        return method(*args, idx=idx, **kwargs)
    return get_full_batch


def _get_batch_method(method, name):
    def get_batch(self, *args, batch_size=1, vr='none', random_state=None,
                  **kwargs):
        rng = check_random_state(random_state)
        if batch_size is None or batch_size == 'all':
            batch_size = self.n_samples

        assert vr in ['none', 'saga'], (
            f"'vr' should be in ['none', 'saga']. Got '{vr}'."
        )
        use_vr = vr != 'none'
        memory, vr_res = None, None

        if use_vr:
            assert batch_size == 1
            vr_res, memory = self.memory.get(name, (None, None))
            # Initialize the memory if it does not exists
            if memory is None:
                all_results = [
                    method(*args, idx=[i], **kwargs)
                    for i in range(self.n_samples)
                ]
                params = all_results[0]
                if isinstance(params, tuple):
                    memory, vr_res = [], []
                    for j in range(len(params)):
                        memory.append(np.array([m[j] for m in all_results]))
                        vr_res.append(memory[j].mean(axis=0))
                else:
                    memory = np.array(all_results)
                    vr_res = memory.mean(axis=0)
                self.memory[name] = (vr_res, memory)
                return vr_res

        idx = rng.choice(
            range(self.n_samples), size=batch_size, replace=False
        )
        res = method(*args, idx=idx, **kwargs)

        if vr == 'none':
            return res

        i = idx[0]
        if isinstance(res, tuple):
            direction = []
            for j in range(len(res)):
                direction.append(res[j] - memory[j][i] + vr_res[j])
                vr_res[j] += (res[j] - memory[j][i]) / self.n_samples
                memory[j][i] = res[j]
        else:
            direction = res - memory[i] + vr_res
            vr_res += (res - memory[i]) / self.n_samples
            memory[i] = res

        return direction

    return get_batch
