import numpy as np

import argparse


parser = argparse.ArgumentParser(
        description='Plot benchmarks results for bilevel optimization.'
    )
parser.add_argument('--n-repetitions', '-r', type=int, default=10,
                    help='# of repetitions.')

args = parser.parse_args()

N_REP = args.n_repetitions

# Store solver specific parameters
SOLVER_DICT = dict(
    amigo=dict(
        name=['AmIGO'],
        batch_size=[64],
        n_inner_step=[10]
    ),
    bsa=dict(
        name=['BSA'],
        batch_size=[64],
        n_inner_step=[10],
        n_hia_step=[10]
    ),
    fsla=dict(
        name=['FSLA'],
        batch_size=[64]
    ),
    mrbo=dict(
        name=['MRBO'],
        batch_size=[64],
        n_hia_step=[10],
        eta=[.5]
    ),
    saba=dict(
        name=['SABA'],
        batch_size=[64]
    ),
    soba=dict(
        name=['SOBA'],
        batch_size=[64, 'full']
    ),
    stocbio=dict(
        name=['StocBiO'],
        batch_size=[64],
        n_inner_step=[10],
        n_shia_steps=[10]
    ),
    sustain=dict(
        name=['SUSTAIN'],
        batch_size=[64],
        n_hia_step=[10],
        eta=[.5]
    ),
    ttsa=dict(
        name=['TTSA'],
        batch_size=[64],
        n_hia_step=[10]
    ),
    sarah=dict(
        name=['BiO-SARAH'],
        batch_size=[64],
        period_frac=[0.25, 0.5, 1., 2., 4., 8.]
    ),
    svrg=dict(
        name=['BiO-SVRG'],
        batch_size=[64],
        period_frac=[0.25, 0.5, 1., 2., 4., 8.]
    ),
)

# Store benchmark specific parameters
BENCH_DICT = dict(
    ijcnn1=dict(
        eval_freq=[2**17],
        PATIENCE=50,
        step_size=np.logspace(-5, 3, 9, base=2),
        # step_size=[1e-4],
        outer_ratio=np.logspace(-2, 1, 7),
        # outer_ratio=[1.],
        dataset='ijcnn1',
        model='logreg',
        n_reg='full',
        reg='exp',
        task='classif',
        n=1500,
        timeout=800,
        numba=True
    ),
    covtype=dict(
        eval_freq=[2**5],
        PATIENCE=12_800,
        step_size=np.logspace(-5, 3, 9, base=2),
        outer_ratio=np.logspace(-2, 1, 6),
        dataset='covtype',
        model='multilogreg',
        n_reg='full',
        reg='exp',
        task='classif',
        n=64000,
        timeout=300,
        numba=False
    ),
    datacleaning0_5=dict(
        eval_freq=[2**5],
        PATIENCE=12_800,
        step_size=np.logspace(-3, 2, 11),
        outer_ratio=np.logspace(-5, 0, 11),
        ratio=[.5],
        dataset='mnist',
        model='None',
        n_reg='full',
        reg='exp',
        task='datacleaning',
        n=64000,
        timeout=720,
        numba=False
    ),
    datacleaning0_7=dict(
        eval_freq=[2**5],
        PATIENCE=12_800,
        step_size=np.logspace(-3, 2, 11),
        outer_ratio=np.logspace(-5, 0, 11),
        ratio=[.7],
        dataset='mnist',
        model='None',
        n_reg='full',
        reg='exp',
        task='datacleaning',
        n=64000,
        timeout=720,
        numba=False
    ),
    datacleaning0_9=dict(
        eval_freq=[2**5],
        PATIENCE=12_800,
        step_size=np.logspace(-3, 2, 11),
        outer_ratio=np.logspace(-5, 0, 11),
        ratio=[.9],
        dataset='mnist',
        model='None',
        n_reg='full',
        reg='exp',
        task='datacleaning',
        n=64000,
        timeout=720,
        numba=False
    ),
    twentynews_binary=dict(
        eval_freq=[2**5],
        PATIENCE=12_800,
        step_size=np.logspace(-5, 3, 9, base=2),
        outer_ratio=np.logspace(-2, 1, 6),
        dataset='20news_binary',
        model='logreg',
        n_reg='1',
        reg='exp',
        task='classif',
        n=64000,
        timeout=120,
        numba=False
    ),
)


OBJECTIVE_DICT = dict(
    (
        benchmark,
        dict((key, BENCH_DICT[benchmark][key])
             for key in ['model', 'n_reg', 'numba', 'reg', 'task'])
    )
    for benchmark in BENCH_DICT
)

DATASET_DICT = dict(
    ijcnn1=["ijcnn1"],
    covtype=["covtype"],
    datacleaning0_5=["mnist[ratio=0.5]"],
    datacleaning0_7=["mnist[ratio=0.7]"],
    datacleaning0_9=["mnist[ratio=0.9]"],
    twentynews_binary=["20news_binary"],
)

for benchmark in BENCH_DICT:
    with open(f"{benchmark}.yml", "x") as f:
        f.write("objective:\n")
        f.write("  - Bilevel Optimization[")
        for key in ['model', 'n_reg', 'numba', 'reg', 'task']:
            f.write(f"{key}={OBJECTIVE_DICT[benchmark][key]}")
            if key != 'task':
                f.write(',')
        f.write("]\n")

        f.write("dataset:\n")
        for d in DATASET_DICT[benchmark]:
            f.write(f"  - {d}\n")

        f.write('solver:\n')
        for s in SOLVER_DICT:
            temp_dict = SOLVER_DICT[s]
            temp_dict.update(
                dict(eval_freq=BENCH_DICT[benchmark]['eval_freq'])
            )
            temp_dict.update(dict(
                step_size=list(BENCH_DICT[benchmark]['step_size'])
                )
            )
            temp_dict.update(dict(
                outer_ratio=list(BENCH_DICT[benchmark]['outer_ratio'])
                )
            )
            f.write(f'  - {temp_dict["name"][0]}[')
            for i, param in enumerate(temp_dict):
                if param != 'name':
                    f.write(f'{param}={temp_dict[param]}')
                    if i != len(temp_dict) - 1:
                        f.write(',')
                    else:
                        f.write(']\n')

        # f.write(f'PATIENCE: {BENCH_DICT[benchmark]["PATIENCE"]}\n')
        f.write(f'n-repetitions: {N_REP}\n')
        f.write(f'max-runs: {BENCH_DICT[benchmark]["n"]}\n')
        f.write(f'timeout: {BENCH_DICT[benchmark]["timeout"] * N_REP}\n')
        f.write(f'output: {benchmark}\n')
