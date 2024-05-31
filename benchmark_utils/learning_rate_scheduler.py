from jax import jit


@jit
def update_lr(state):
    """Update the learning rate according to a scheduler."""
    lr = state['constants'] / ((state['i_step'] + 1) ** state['exponents'])
    state['i_step'] += 1
    return lr, state


def init_lr_scheduler(constants, exponents):
    """Initialize a state of the learning rate scheduler."""
    return {
        'i_step': 0,
        'constants': constants,
        'exponents': exponents
    }
