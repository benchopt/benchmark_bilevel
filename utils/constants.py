import numpy as np

EVAL_FREQ = 2 ** 17
PATIENCE = 100

STEP_SIZES = [1e-1]
OUTER_RATIOS = [10]
BATCH_SIZES = [1, 32]

# Number of inner loop steps
N_INNER_STEPS = [10]
N_HIA_STEPS = [10]

# STORM momentum parameter
ETA = [0.5]

# Get a random seed
MAX_SEED = 2 ** 32 - 1
RANDOM_STATE = np.random.randint(MAX_SEED)
print(f"SEED = {RANDOM_STATE}")
