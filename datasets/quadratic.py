from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils import oracles


class Dataset(BaseDataset):

    name = "quadratic"

    parameters = {
        'oracle': ['quadratic'],
        'L_inner': [1.],
        'mu_inner': [.1],
        'L_outer': [1.],
        'random_state': [2442],
        'n_samples_inner': [1000],
        'n_samples_outer': [1000],
        'dim_inner': [100],
        'dim_outer': [100],
    }

    def get_data(self):
        rng = np.random.RandomState(self.random_state)

        hess_inner_list = []

        for i in range(self.n_samples_inner):
            A = rng.randn(self.dim_inner, self.dim_inner)
            A = A.T @ A
            _, U = np.linalg.eigh(A)
            D = rng.uniform(self.mu_inner, self.L_inner, self.dim_inner)
            D = np.diag(D)
            A = U @ D @ U.T
            hess_inner_list.append(A)

        cross_list = []
        for i in range(self.n_samples_inner):
            A = rng.randn(self.dim_outer, self.dim_inner)
            cross_list.append(A)

        linear_inner_list = []
        for i in range(self.n_samples_inner):
            A = rng.randn(self.dim_inner)
            linear_inner_list.append(A)

        hess_outer_list = []

        for j in range(self.n_samples_outer):
            A = rng.randn(self.dim_outer, self.dim_outer)
            A = A.T @ A
            _, U = np.linalg.eigh(A)
            D = rng.uniform(0, self.L_outer, self.dim_outer)
            D = np.diag(D)
            A = U @ D @ U.T
            hess_outer_list.append(A)

        linear_outer_list = []
        for j in range(self.n_samples_outer):
            A = rng.randn(self.dim_inner)
            linear_outer_list.append(A)


        def get_inner_oracle(framework="none", get_full_batch=False):
            oracle = oracles.QuadraticOracle(
                hess_inner_list, hess_outer_list, cross_list,
                linear_inner_list, self.L_inner, self.mu_inner, self.L_outer
            )
            return oracle.get_framework(framework=framework,
                                        get_full_batch=get_full_batch)

        def get_outer_oracle(framework="none", get_full_batch=False):
            oracle = oracles.QuadraticOracle(
                hess_inner_list, hess_outer_list, cross_list,
                linear_inner_list, self.L_inner, self.mu_inner, self.L_outer
            )
            return oracle.get_framework(framework=framework,
                                        get_full_batch=get_full_batch)

        def metrics(inner_var, outer_var):
            f_outer = get_outer_oracle()
            hess_inner = np.mean(hess_inner_list, axis=0)
            cross = np.mean(cross_list, axis=0)
            linear_inner = np.mean(linear_inner_list, axis=0)
            linear_outer = np.mean(linear_outer_list, axis=0)
            inner_sol = np.linalg.solve(
                hess_inner,
                - linear_inner - outer_var @ cross
            )
            v_sol = - np.linalg.solve(
                hess_inner,
                f_outer.get_grad_inner_var(inner_sol, outer_var)
            )
            grad_value = f_outer.get_grad_outer_var(inner_sol, outer_var)
            grad_value += cross @ v_sol
            return dict(
                func=float(f_outer.get_value(inner_sol, outer_var)),
                value=np.linalg.norm(grad_value)**2,
                inner_distance=np.linalg.norm(inner_sol - inner_var)**2,

            )

        data = dict(
            get_inner_oracle=get_inner_oracle,
            get_outer_oracle=get_outer_oracle,
            oracle='quadratic',
            metrics=metrics,
            n_reg=None,
        )
        return data
