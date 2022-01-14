import numpy as np

EVAL_FREQ = 2 ** 15

STEP_SIZES = [1e-2]
OUTER_RATIOS = [2]
BATCH_SIZES = [32]

# Number of inner loop steps
N_INNER_STEPS = [10]
N_HIA_STEPS = [10]


# Get a random seed
MAX_SEED = 2 ** 32 - 1
RANDOM_STATE = np.random.randint(MAX_SEED)
print(f"SEED = {RANDOM_STATE}")
