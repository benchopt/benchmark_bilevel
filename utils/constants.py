import numpy as np

EVAL_FREQ = 1
PATIENCE = 150

STEP_SIZES = [1e-2]
OUTER_RATIOS = [1e-1]
BATCH_SIZES = [64]

# Number of inner loop steps
N_INNER_STEPS = [10]
N_HIA_STEPS = [10]

# STORM momentum parameter
ETA = [0.5]

# Get a random seed
MAX_SEED = 2 ** 32 - 1
RANDOM_STATE = np.random.randint(MAX_SEED)
print(f"SEED = {RANDOM_STATE}")
