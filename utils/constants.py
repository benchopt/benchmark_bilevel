import numpy as np

EVAL_FREQ = 2 ** 7
PATIENCE = 50

STEP_SIZES = np.logspace(-5, 3, 9, base=2)
OUTER_RATIOS = np.logspace(-2, 1, 7)
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
