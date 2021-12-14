Bi-level Optimization Benchmark
===============================
|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver for bi-level optimization:

.. math::

    \min_{x} f(x, z^*(x)) \quad with \quad z^*(x) = \argmin_z g(x, z)

where n (or n_samples) stands for the number of samples, p (or n_features) stands for the number of features and

.. math::

 X = [x_1^\top, \dots, x_n^\top]^\top \in \mathbb{R}^{n \times p}

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_bilevel
   $ benchopt run benchmark_bilevel

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_bilevel -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_bilevel/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_bilevel/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
