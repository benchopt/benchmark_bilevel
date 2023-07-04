Bilevel Optimization Benchmark
===============================
|Build Status| |Python 3.6+|

*Results can be consulted on https://benchopt.github.io/results/benchmark_bilevel.html*

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solvers for bilevel optimization:

.. math::

    \min_{x} f(x, z^*(x)) \quad with \quad z^*(x) = \arg\min_z g(x, z), 

where :math:`g` and :math:`f` are two functions of two variables.

Different problems
------------------

This benchmark currently implements two bilevel optimization problems

1 - Regularization selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this problem, the inner function :math:`g` is defined by 

.. math::

   g(x, z) = \frac{1}{n} \sum_{i=1}^{n} \ell(d_i; z) + \mathcal{R}(x, z)

where :math:`d_1, \dots, d_n` are training data samples, :math:`z` are the parameters of the machine learning model, and the loss function :math:`\ell` measures how well the model parameters :math:`z` predict the data :math:`d_i`.
There is also a regularization :math:`\mathcal{R}` that is parametrized by the regularization strengths :math:`x`, which aims at promoting a certain structure on the parameters :math:`z`.

The outer function :math:`f` is defined as the unregularized loss on unseen data 

.. math::

   f(x, z) = \frac{1}{m} \sum_{j=1}^{m} \ell(d'_j; z)

where the :math:`d'_1, \dots, d'_m` are new samples from the same dataset as above.

There are currently two datasets for this regularization selection problem.

Covtype
+++++++

*Homepage : https://archive.ics.uci.edu/dataset/31/covertype*

This is a logistic regression problem, where the data is of the form :math:`d_i = (a_i, y_i)` with  :math:`a_i\in\mathbb{R}^p` are the features and :math:`y_i=\pm1` is the binary target.
For this problem, the loss is :math:`\ell(d_i, z) = \log(1+\exp(-y_i a_i^T z))`, and the regularization is simply given by :math:`\mathcal{R}(x, z) = \frac12\sum_{j=1}^p\exp(x_j)z_j^2`: each coefficient in :math:`z` is independently regularized with the strength :math:`\exp(x_j)`.

Ijcnn1
++++++

*Homepage : https://www.openml.org/search?type=data&sort=runs&id=1575&status=active*

This is a multicalss logistic regression problem, where the data is of the form :math:`d_i = (a_i, y_i)` with  :math:`a_i\in\mathbb{R}^p` are the features and :math:`y_i\in \{1,\dots, k\}` is the integer target, with k the number of classes.
For this problem, the loss is :math:`\ell(d_i, z) = \mathrm{CrossEntropy}(za_i, y_i)` where :math:`z` is now a k x p matrix. The regularization is given by :math:`\mathcal{R}(x, z) = \frac12\sum_{j=1}^k\exp(x_j)\|z_j\|^2`: each line in :math:`z` is independently regularized with the strength :math:`\exp(x_j)`.


2 - Hyper data cleaning
^^^^^^^^^^^^^^^^^^^^^^^




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
