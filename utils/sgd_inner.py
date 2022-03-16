# from numba import njit

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    joint_shia = import_ctx.import_from('hessian_approximation', 'joint_shia')


# @njit
def sgd_inner(inner_oracle, inner_var, outer_var, step_size, inner_sampler,
              n_inner_step):

    for i in range(n_inner_step):
        inner_slice, _ = inner_sampler.get_batch()
        grad_inner = inner_oracle.grad_inner_var(
            inner_var, outer_var, inner_slice
        )
        inner_var -= step_size * grad_inner

    return inner_var


# @njit
def sgd_inner_vrbo(inner_oracle, outer_oracle, inner_var, outer_var, inner_lr,
                   inner_sampler, outer_sampler, n_inner_step, memory_inner,
                   memory_outer, n_hia_step, hia_lr):

    for i in range(n_inner_step):
        # Step.4.k.1 - Update direction for z
        slice_inner, _ = inner_sampler.get_batch()
        grad_inner_var = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice_inner
        )
        grad_inner_var_old = inner_oracle.grad_inner_var(
            memory_inner[0], memory_outer[0], slice_inner
        )
        memory_inner[1] += grad_inner_var - grad_inner_var_old

        # Step.4.k.2 - Update direction for x
        slice_outer, _ = outer_sampler.get_batch()
        grad_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        grad_outer_old, impl_grad_old = outer_oracle.grad(
            memory_inner[0], memory_outer[0], slice_outer
        )
        ihvp, ihvp_old = joint_shia(
            inner_oracle, inner_var, outer_var, grad_outer,
            memory_inner[0], memory_outer[0], grad_outer_old,
            inner_sampler, n_hia_step, hia_lr
        )
        impl_grad -= inner_oracle.cross(
            inner_var, outer_var, ihvp, slice_inner
        )
        impl_grad_old -= inner_oracle.cross(
            memory_inner[0], memory_outer[0], ihvp_old, slice_inner
        )
        memory_outer[1] += impl_grad - impl_grad_old

        # Step.4.k.3 - update the inner variable and memory
        memory_inner[0] = inner_var
        inner_var -= inner_lr * memory_inner[1]

        # Step.4.k.4 - project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

    return inner_var, outer_var, memory_inner, memory_outer
