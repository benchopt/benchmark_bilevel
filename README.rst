Bilevel Optimization Benchmark
===============================
|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver for bilevel optimization:

.. math::

    \min_{x} f(x, z^*(x)) \quad with \quad z^*(x) = \argmin_z g(x, z)

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

You can also use config files to setup the benchmark run:

.. code-block::

   $ benchopt run benchmark_bilevel --config config/X.yml

where `X.yml` is a config file. See https://benchopt.github.io/index.html#run-a-benchmark for an example of config file. This will possibly launch a huge grid search. When avalaible, you can rather use the file `X_best_params.yml` in order to lauch an experiment with a single set of parameters for each solver.

Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

If you use this benchmark in your research project, please cite the following paper:

.. code-block::

   @inproceedings{saba,
      title = {A Framework for Bilevel Optimization That Enables Stochastic and Global Variance Reduction Algorithms},
      booktitle = {Advances in {{Neural Information Processing Systems}} ({{NeurIPS}})},
      author = {Dagr{\'e}ou, Mathieu and Ablin, Pierre and Vaiter, Samuel and Moreau, Thomas},
      year = {2022}
   }


.. |Build Status| image:: https://github.com/benchopt/benchmark_bilevel/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_bilevel/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
