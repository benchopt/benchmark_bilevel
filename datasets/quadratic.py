from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils import oracles


class Dataset(BaseDataset):

    name = "quadratic"

    parameters = {
        'oracle': ['quadratic'],
        'L_inner_inner': [1.],
        "L_inner_outer": [1.],
        'mu_inner': [.1],
        'L_outer_inner': [1.],
        "L_outer_outer": [1.],
        'L_cross_inner': [1.],
        "L_cross_outer": [1.],
        'random_state': [2442],
        'n_samples_inner': [1024],
        'n_samples_outer': [1024],
        'dim_inner': [100],
        'dim_outer': [100],
    }

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        inner_seed, outer_seed = rng.randint(2**31-1, size=2)

        f_inner = oracles.QuadraticOracle(
            self.n_samples_inner, self.dim_inner, self.dim_outer,
            self.L_inner_inner, self.L_inner_outer, self.L_cross_inner,
            self.mu_inner,
            random_state=inner_seed
        )
        f_outer = oracles.QuadraticOracle(
            self.n_samples_outer, self.dim_inner, self.dim_outer,
            self.L_outer_inner, self.L_outer_outer, self.L_cross_outer,
            self.mu_inner,
            random_state=outer_seed
        )
        hess_inner = f_inner.hess_inner_full
        cross = f_inner.cross_mat_full
        linear_inner = f_inner.linear_inner_full

        def get_inner_oracle(framework="none", get_full_batch=False):
            return f_inner.get_framework(
                framework=framework, get_full_batch=get_full_batch
            )

        def get_outer_oracle(framework="none", get_full_batch=False):
            return f_outer.get_framework(
                framework=framework, get_full_batch=get_full_batch
            )

        def metrics(inner_var, outer_var):
            inner_sol = np.linalg.solve(
                hess_inner,
                - linear_inner - cross.T @ outer_var
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
