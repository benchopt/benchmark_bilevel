import numpy as np

EVAL_FREQ = 2 ** 5
PATIENCE = 12_800
BATCH_SIZES = [64]

STEP_SIZES = np.logspace(-3, 0, 7)
OUTER_RATIOS = np.logspace(-5, -2, 7)

# Number of inner loop steps
N_INNER_STEPS = [10]
N_HIA_STEPS = [10]

# STORM momentum parameter
ETA = [0.5]

# Get a random seed
MAX_SEED = 2 ** 32 - 1
# RANDOM_STATE = np.random.randint(MAX_SEED)
RANDOM_STATE = None
print(f"SEED = {RANDOM_STATE}")
