import numpy as np

EVAL_FREQ = 2 ** 4
PATIENCE = 30000

STEP_SIZES = np.logspace(-5, -1, 4)
OUTER_RATIOS = np.logspace(-3, 1, 4)
BATCH_SIZES = [64]

# Number of inner loop steps
N_INNER_STEPS = [10, 50]
N_HIA_STEPS = [3, 10]

# STORM momentum parameter
ETA = [0.5]

# Get a random seed
MAX_SEED = 2 ** 32 - 1
RANDOM_STATE = 666
print(f"SEED = {RANDOM_STATE}")
