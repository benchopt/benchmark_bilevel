import jax
from functools import partial


def sgd_inner(inner_oracle, inner_var, outer_var, step_size, sampler=None,
              n_steps=1):

    for i in range(n_steps):
        inner_slice, _ = sampler.get_batch()
        grad_inner = inner_oracle.grad_inner_var(
            inner_var, outer_var, inner_slice
        )
        inner_var -= step_size * grad_inner

    return inner_var


@partial(jax.jit, static_argnums=(0, ), static_argnames=('sampler', 'n_steps'))
def sgd_inner_jax(grad_inner, inner_var, outer_var, state_sampler, step_size,
                  sampler=None, n_steps=1):

    def iter(i, args):
        start, args[0] = sampler(**args[0])
        args[1] -= step_size * grad_inner(args[1], outer_var, start)
        return args
    res = jax.lax.fori_loop(0, n_steps, iter, [state_sampler, inner_var])

    return res[1], res[0]


def sgd_inner_vrbo(joint_shia, inner_oracle, outer_oracle, inner_var,
                   outer_var, inner_lr, inner_sampler, outer_sampler,
                   n_inner_steps, memory_inner, memory_outer, n_shia_steps,
                   hia_lr):

    for i in range(n_inner_steps):
        # Step.4.k.1 - Update direction for z
        slice_inner, _ = inner_sampler.get_batch()
        grad_inner_var = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice_inner
        )
        grad_inner_var_old = inner_oracle.grad_inner_var(
            memory_inner[0], outer_var, slice_inner
        )
        memory_inner[1] += grad_inner_var - grad_inner_var_old

        # Step.4.k.2 - Update direction for x
        slice_outer, _ = outer_sampler.get_batch()
        grad_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        grad_outer_old, impl_grad_old = outer_oracle.grad(
            memory_inner[0], outer_var, slice_outer
        )
        ihvp, ihvp_old = joint_shia(
            inner_oracle, inner_var, outer_var, grad_outer,
            memory_inner[0], outer_var, grad_outer_old,
            hia_lr, sampler=inner_sampler, n_steps=n_shia_steps
        )
        impl_grad -= inner_oracle.cross(
            inner_var, outer_var, ihvp, slice_inner
        )
        impl_grad_old -= inner_oracle.cross(
            memory_inner[0], outer_var, ihvp_old, slice_inner
        )
        memory_outer[1] += impl_grad - impl_grad_old

        # Step.4.k.3 - update the inner variable and memory
        memory_inner[0] = inner_var
        inner_var -= inner_lr * memory_inner[1]

        # Step.4.k.4 - project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

    return inner_var, outer_var, memory_inner, memory_outer
