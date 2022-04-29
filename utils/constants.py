import numpy as np

EVAL_FREQ = 2 ** 4
PATIENCE = 3000

STEP_SIZES = [.1]
OUTER_RATIOS = [.01]
BATCH_SIZES = ['full']

# Number of inner loop steps
N_INNER_STEPS = [10, 50]
N_HIA_STEPS = [3, 10]

# STORM momentum parameter
ETA = [0.5]

# Get a random seed
MAX_SEED = 2 ** 32 - 1
RANDOM_STATE = 666
print(f"SEED = {RANDOM_STATE}")
