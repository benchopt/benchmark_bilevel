Bilevel Optimization Benchmark
===============================
[![test](https://github.com/benchopt/benchmark_bilevel/workflows/Tests/badge.svg)](https://github.com/benchopt/benchmark_bilevel/actions)
[![python](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/release/python-360/)

*Results can be consulted on https://benchopt.github.io/results/benchmark_bilevel.html*

BenchOpt is a package to simplify, to make more transparent, and
reproducible the comparison of optimization algorithms.
This benchmark is dedicated to solvers for bilevel optimization:

$$\min_{x} f(x, z^* (x)) \quad \text{with} \quad z^*(x) = \arg\min_z g(x, z),$$

where $g$ and $f$ are two functions of two variables.

Different problems
------------------

This benchmark implements three bilevel optimization problems: quadratic problem, regularization selection, and data cleaning.

### 1 - Simulated quadratic bilevel problem


In this problem, the inner and the outer functions are quadratic functions defined on $\mathbb{R}^{d\times p}$

$$g(x, z) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2} z^\top A_i z + \frac{1}{2} x^\top B_i x + x^\top C_i z + a_i^\top z + b_i^\top x$$

and

$$f(x, z) = \frac{1}{m} \sum_{j=1}^m \frac{1}{2} z^\top F_j z + \frac{1}{2} x^\top H_j x + x^\top K_j z + f_j^\top z + h_j^\top x$$

where $A_i, F_j$ are symmetric positive definite matrices of size $p\times p$, $B_i, F_j$ are symmetric positive definite matrices of size $d\times d$, $C_i, K_j$ are matrices of size $d\times p$, $a_i$, $f_j$ are vectors of size $d$, and $b_i, h_j$ are vectors of size $p$.

The matrices $A_i, B_i, F_j, H_j$ are randomly generated such that the eigenvalues of $\frac1n\sum_i A_i$ are between ``mu_inner``, and ``L_inner_inner``, the eigenvalues of $\frac1n\sum_i B_i$ are between ``mu_inner``, and ``L_inner_outer``, the eigenvalues of $\frac1m\sum_j F_j$ are between ``mu_inner``, and ``L_outer_inner``, and the eigenvalues of $\frac1m\sum_j H_j$ are between ``mu_inner``, and ``L_outer_outer``.

The matrices $C_i, K_j$ are generated randomly such that the spectral norm of $\frac1n\sum_i C_i$ is lower than ``L_cross_inner``, and the spectral norm of $\frac1m\sum_j K_j$ is lower than ``L_cross_outer``.

Note that in this setting, the solution of the inner problem is a linear system.
As the full batch inner and outer functions can be computed efficiently with the average Hessian matrices, the value function is evaluated in closed form. 


### 2 - Regularization selection

In this problem, the inner function $g$ is defined by 


$$g(x, z) = \frac{1}{n} \sum_{i=1}^{n} \ell(d_i; z) + \mathcal{R}(x, z)$$

where $d_1, \dots, d_n$ are training data samples, $z$ are the parameters of the machine learning model, and the loss function $\ell$ measures how well the model parameters $z$ predict the data $d_i$.
There is also a regularization $\mathcal{R}$ that is parametrized by the regularization strengths $x$, which aims at promoting a certain structure on the parameters $z$.

The outer function $f$ is defined as the unregularized loss on unseen data

$$f(x, z) = \frac{1}{m} \sum_{j=1}^{m} \ell(d'_j; z)$$

where the $d'_1, \dots, d'_m$ are new samples from the same dataset as above.

There are currently two datasets for this regularization selection problem.

#### Covtype - [*Homepage*](https://archive.ics.uci.edu/dataset/31/covertype*)

This is a logistic regression problem, where the data have the form $d_i = (a_i, y_i)$ with $a_i\in\mathbb{R}^p$ the features and $y_i=\pm1$ the binary target.
For this problem, the loss is $\ell(d_i, z) = \log(1+\exp(-y_i a_i^T z))$, and the regularization is simply given by
$$\mathcal{R}(x, z) = \frac12\sum_{j=1}^p\exp(x_j)z_j^2,$$
each coefficient in $z$ is independently regularized with the strength $\exp(x_j)$.

#### Ijcnn1 - [*Homepage*](https://www.openml.org/search?type=data&sort=runs&id=1575&status=active)

This is a multiclass logistic regression problem, where the data is of the form $d_i = (a_i, y_i)$ with  $a_i\in\mathbb{R}^p$ are the features and $y_i\in \{1,\dots, k\}$ is the integer target, with k the number of classes.
For this problem, the loss is $\ell(d_i, z) = \text{CrossEntropy}(za_i, y_i)$ where $z$ is now a k x p matrix. The regularization is given by 
$$\mathcal{R}(x, z) = \frac12\sum_{j=1}^k\exp(x_j)\|z_j\|^2,$$
each line in $z$ is independently regularized with the strength $\exp(x_j)$.


### 3 - Data cleaning

This problem was first introduced by [Franceschi et al., 2017](https://arxiv.org/abs/1703.01785).
In this problem, the data is the MNIST dataset.
The training set has been corrupted: with a probability $p$, the label of the image $`y\in\{1,\dots,10\}`$ is replaced by another random label between 1 and 10.
We do not know beforehand which data has been corrupted.
We have a clean testing set, which has not been corrupted.
The goal is to fit a model on the corrupted training data that has good performances on the test set.
To do so, a set of weights -- one per train sample -- is learned as well as the model parameters.
Ideally, we would want a weight of 0 for data that has been corrupted and a weight of 1 for uncorrupted data.
The problem is cast as a bilevel problem with $g$ given by 

$$g(x, z) =\frac1n \sum_{i=1}^n \sigma(x_i)\ell(d_i, z) + \frac C 2 \|z\|^2$$

where the $d_i$ are the corrupted training data, $\ell$ is the loss of a CNN parameterized by $z$, $\sigma$ is a sigmoid function, and C is a small regularization constant.
Here the outer variable $x$ is a vector of dimension $n$, and the weight of data $i$ is given by $\sigma(x_i)$.
The test function is

$$f(x, z) =\frac1m \sum_{j=1}^n \ell(d'_j, z)$$

where the $d_j$ are uncorrupted testing data.

Install
--------

This benchmark can be run using the following commands:

```bash
   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_bilevel
   $ benchopt run benchmark_bilevel
```

Apart from the problem, options can be passed to ``benchopt run`` to restrict the benchmarks to some solvers or datasets, e.g.:

```bash
   $ benchopt run benchmark_bilevel -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10
````

You can also use config files to set the benchmark run:

```bash
   $ benchopt run benchmark_bilevel --config config/X.yml
```

where ``X.yml`` is a config file. See https://benchopt.github.io/index.html#run-a-benchmark for an example of a config file. This will launch a huge grid search. When available, you can rather use the file ``X_best_params.yml`` to launch an experiment with a single set of parameters for each solver.

Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

### How to contribute to the benchmark?

If you want to add a solver or a new problem, you are welcome to open an issue or submit a pull request!  

#### 1 - How to add a new solver?

Each solver derives from the [`benchopt.BaseSolver` class](https://benchopt.github.io/user_guide/generated/benchopt.BaseSolver.html) in the [solvers](solvers) folder. The solvers are separated among the stochastic JAX solvers and the others:
* Stochastic Jax solver: these solvers inherit from the [`StochasticJaxSolver` class](benchmark_utils/stochastic_jax_solver.py) see the detailed explanations in the [template stochastic solver](solvers/template_stochastic_solver.py).
* Other solver: see the detailed explanation in the [Benchopt documentation](https://benchopt.github.io/tutorials/add_solver.html). An example is provided in the [template solver](solvers/template_solver.py).

#### 2 - How to add a new problem?

In this benchmark, each problem is defined by a [Dataset class](https://benchopt.github.io/user_guide/generated/benchopt.BaseDataset.html) in the [datasets](datasets) folder. A [template](datasets/template_dataset.py) is provided.

Cite
----

If you use this benchmark in your research project, please cite the following paper:

```
   @inproceedings{dagreou2022,
      title = {A Framework for Bilevel Optimization That Enables Stochastic and Global Variance Reduction Algorithms},
      booktitle = {Advances in {{Neural Information Processing Systems}} ({{NeurIPS}})},
      author = {Dagr{\'e}ou, Mathieu and Ablin, Pierre and Vaiter, Samuel and Moreau, Thomas},
      year = {2022}
   }
```
