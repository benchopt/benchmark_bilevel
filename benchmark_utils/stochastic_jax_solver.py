from abc import ABC, abstractmethod

from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils import constants
    from benchmark_utils.minibatch_sampler import init_sampler

    import jax
    from functools import partial


class StochasticJaxSolver(BaseSolver, ABC):
    """Base class for stochastic solvers using JAX.

    Attributes
    ----------
    f_inner, f_outer: callable
        Inner and outer objective function for the bilevel optimization
        problem. Should take as input:
            * inner_var: array-like, shape (dim_inner,)
            * outer_var: array-like, shape (dim_outer,)
            * start: int, the starting index of the minibatch
            * batch_size: int, the size of the minibatch

    n_inner_samples, n_outer_samples: int
        Number of samples to draw for the inner and outer objective functions.

    inner_var0, outer_var0: array-like, shape (dim_inner,) (dim_outer,)

    f_inner_fb, f_outer_fb: callable
        Full batch version of f_inner and f_outer. Should take as input:
            * inner_var: array-like, shape (dim_inner,)
            * outer_var: array-like, shape (dim_outer,)

    batch_size: int or 'full'
        Size of the minibatch to use. If 'full', the full batch is used.

    random_state: int
        Random seed to use.

    eval_freq: int
        Frequency at which to evaluate the objective function.

    stop_val: int
        Current iteration of the solver.

    stopping_criterion: SufficientProgressCriterion
        Stopping criterion for the solver.

    one_epoch: callable
        Function that runs one epoch of the solver.

    inner_var, outer_var: array-like, shape (dim_inner,) (dim_outer,)
        Current values of the inner and outer variables.

    state_inner_sampler, state_outer_sampler: array-like
        State of the random number generator for the inner and outer sampler.

    """

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'eval_freq': [128],
        'random_state': [1],
    }

    requirements = ["jax", "jaxlib"]

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_inner, f_outer, n_inner_samples, n_outer_samples,
                      inner_var0, outer_var0):
        """Set the problem to solve.

        Parameters
        ----------
        f_inner, f_outer: callable
            Inner and outer objective function for the bilevel optimization
            problem. Should take as input:
                * inner_var: array-like, shape (dim_inner,)
                * outer_var: array-like, shape (dim_outer,)
                * start: int, the starting index of the minibatch
                * batch_size: int, the size of the minibatch

        n_inner_samples, n_outer_samples: int
            Number of samples to draw for the inner and outer objective
            functions.

        inner_var0, outer_var0: array-like, shape (dim_inner,) (dim_outer,)

        Attributes
        ----------
        f_inner, f_outer: callable
            Inner and outer objective function for the bilevel optimization
            problem.

        n_inner_samples, n_outer_samples: int
            Number of samples to draw for the inner and outer objective
            functions.

        inner_var0, outer_var0: array-like, shape (dim_inner,) (dim_outer,)

        batch_size_inner, batch_size_outer: int
            Size of the minibatch to use for the inner and outer objective
            functions.

        state_inner_sampler, state_outer_sampler: dict
            State of the minibatch samplers for the inner and outer objectives.

        one_epoch: callable
            Jitted function that runs the solver for one epoch. One epoch is
            defined as `eval_freq` iterations of the solver.
        """

        self.f_inner = f_inner
        self.f_outer = f_outer
        self.n_inner_samples = n_inner_samples
        self.n_outer_samples = n_outer_samples

        if self.batch_size == "full":
            self.batch_size_inner = n_inner_samples
            self.batch_size_outer = n_outer_samples
        else:
            self.batch_size_inner = self.batch_size
            self.batch_size_outer = self.batch_size

        self.f_inner = partial(self.f_inner, batch_size=self.batch_size_inner)
        self.f_outer = partial(self.f_outer, batch_size=self.batch_size_outer)

        keys = jax.random.split(jax.random.PRNGKey(self.random_state), 2)

        inner_sampler, self.state_inner_sampler = init_sampler(
            n_samples=n_inner_samples, batch_size=self.batch_size_inner,
            key=keys[0]
        )
        outer_sampler, self.state_outer_sampler = init_sampler(
            n_samples=n_outer_samples, batch_size=self.batch_size_outer,
            key=keys[1]
        )
        self.one_epoch = self.get_one_epoch_jitted(
            inner_sampler, outer_sampler
        )

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        # warmup
        self.run_once(2)

    @abstractmethod
    def init(self):
        """Init the carry of the stochastic algorithm.

        The carry should at least contain `inner_var` and `outer_var`.
        """
        ...

    @abstractmethod
    def get_step(self, inner_sampler, outer_sampler):
        """Returns a function that compute one iteration of the stochastic
        algorithm.
        """
        ...

    def run(self, callback):
        """Run the solver.
        The callback is called every `self.eval_freq`iterations."""
        carry = self.init()

        # Start algorithm
        while callback():
            carry = self.one_epoch(carry, self.eval_freq)
            self.inner_var = carry["inner_var"]
            self.outer_var = carry["outer_var"]

    def get_result(self):
        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var
        )

    def get_one_epoch_jitted(self, inner_sampler, outer_sampler):
        step = self.get_step(inner_sampler, outer_sampler)

        def one_epoch(carry, eval_freq):
            carry, _ = jax.lax.scan(
                step, init=carry, xs=None,
                length=eval_freq,
            )
            return carry

        return jax.jit(
            one_epoch, static_argnums=(1,)
        )
