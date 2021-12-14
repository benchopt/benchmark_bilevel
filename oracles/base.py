from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils import check_random_state


METHODS = [
    'value', 'grad_inner_var', 'grad_outer_var', 'grad',
    'cross', 'hvp', 'inverse_hessian_vector_prod', 'oracles',
    'inner_var_star'
]
GET_METHODS = [f"get_{m}" for m in METHODS]
GET_BATCH_METHODS = [f"get_batch_{m}" for m in METHODS]


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

    - `inverse_hessian_vector_prod(inner_var, outer_var, v, idx): should
      return the product between the inverse Hessian (with respect to the inner
      variable) at inner_var and outer_var estimated on the indices contained
      in idx and a vector v.

    Note that the batch size should be defined in __init__.
    """
    # Shape of the variable for the considered problem
    variable_shape = None

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
    def inverse_hessian_vector_prod(self, inner_var, outer_var, v, idx):
        pass

    def grad(self, inner_var, outer_var, idx):
        return self.grad_inner_var(inner_var, outer_var, idx), \
            self.grad_outer_var(inner_var, outer_var, idx)

    def oracles(self, inner_var, outer_var, v, idx):
        return self.value(inner_var, outer_var, idx), \
            self.grad_inner_var(inner_var, outer_var, idx), \
            self.cross(inner_var, outer_var, v, idx), \
            self.hvp(inner_var, outer_var, v, idx)

    def inner_var_star(self, outer_var, idx):
        var_shape_flat = np.prod(self.variable_shape)

        def func(inner_var):
            inner_var = inner_var.reshape(*self.variable_shape)
            return self.value(inner_var, outer_var, idx)

        def fprime(inner_var):
            inner_var = inner_var.reshape(*self.variable_shape)
            return self.grad_inner_var(inner_var, outer_var, idx)

        inner_var_star, _, _ = fmin_l_bfgs_b(
            func, np.zeros(var_shape_flat), fprime=fprime
        )
        return inner_var_star

    def get_value_function(self, outer_var):
        idx = np.arange(self.n_samples)
        inner_var_star = self.inner_var_star(outer_var, idx=idx)
        return self.value(inner_var_star, outer_var, idx=idx)

    def set_batch_size(self, batch_size):
        if batch_size == 'all':
            self.batch_size = self.n_samples
        else:
            self.batch_size = batch_size

    def __getattribute__(self, name):
        # construct get_* and get_batch_* for all methods in this list:
        if name in GET_METHODS:
            method_name = name.replace('get_', '')

            def get_m(self, *args, **kargs):
                idx = np.arange(self.n_samples)
                return getattr(self, method_name)(*args, idx)
            return get_m.__get__(self, BaseOracle)

        if name in GET_BATCH_METHODS:
            method_name = name.replace('get_batch', '')

            def get_batch_m(self, *args, random_state=None, **kargs):
                rng = check_random_state(random_state)
                idx = rng.choice(
                    range(self.n_samples), size=self.batch_size, replace=False
                )
                return getattr(self, method_name)(*args, idx)
            return get_batch_m.__get__(self, BaseOracle)

        return super().__getattribute__(name)
