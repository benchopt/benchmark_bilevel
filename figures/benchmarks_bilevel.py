from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


FILE_NAME = Path(__file__).with_suffix('')

# Utils to get common STYLES object and setup matplotlib
# for all plots

mpl.rcParams.update({
    'font.size': 10,
    'legend.fontsize': 'small',
    'axes.labelsize': 'small',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small'
})


STYLES = {
    '*': dict(lw=2),

    # stochastic solvers
    'saba': dict(color='C0', label='SABA', zorder=10, lw=3),
    'soba': dict(color='C1', label='SOBA', zorder=10, lw=3),
    'fsla': dict(color='C2', label='FSLA'),

    # HIA solvers
    'mrbo': dict(color='C5', linestyle=':', label='MRBO'),
    'sustain': dict(color='C6', linestyle=':', label='SUSTAIN'),
    'ttsa': dict(color='C8', linestyle=':', label='TTSA'),

    # Two loops solvers
    'amigo': dict(color='C3', linestyle='--', label='AmIGO'),
    'stocbio': dict(color='C4', linestyle='--', label='StocBiO'),
    'bsa': dict(color='C7', linestyle='--', label='BSA'),
}

DEFAULT_WIDTH = 3.25
DEFAULT_DOUBLE_WIDTH = 6.75
DEFAULT_HEIGHT = 2.5


if __name__ == "__main__":

    BENCHMARKS = [
        # ("benchmark_ijcnn1.csv", 'objective_value_func', (0, 500), 1e-4),
        ("run_20_01_ijcnn1.csv", 'objective_value_func', (0, 500), 1e-4),
        # ("benchmark_datacleaning.csv", 'objective_value', (0, 250), 1e-3),
        # ("run_mdagreou.csv",
        ("datacleaning.csv", 'objective_value', (0, 40), None),
    ]

    for fname, metric, xlim, eps in BENCHMARKS:
        fname = FILE_NAME.parent / fname
        fname_reduced = fname.with_suffix('.parquet')

        if not fname_reduced.exists():
            df = pd.read_csv(fname)

            # normalize names
            df['solver'] = df['solver_name'].apply(
                lambda x: x.split('[')[0].lower()
            )

            # Select curve that reach the lowest point
            to_plot = (
                df  # .query('stop_val <= 100')
                .groupby(['solver', 'solver_name', 'stop_val']).median()
                .reset_index().sort_values(metric)
                .groupby('solver').first()['solver_name']
            )
            to_plot = [to_plot[p] for p in STYLES if p in to_plot]
            df = df.query("solver_name in @to_plot")
            df.to_parquet(fname_reduced)

        df = pd.read_parquet(fname_reduced)
        solvers = [s for s in STYLES if s in df['solver'].values]
        to_plot = df.set_index('solver').loc[solvers, 'solver_name']
        to_plot = to_plot.unique()

        print("Chosen parameters:")
        for s in to_plot:
            print(f"- {s}")

        legend_ratio = 0.1
        fig = plt.figure(
            figsize=(DEFAULT_WIDTH, DEFAULT_HEIGHT * (1 + legend_ratio))
        )
        gs = plt.GridSpec(
            2, 1, height_ratios=[legend_ratio, 1], top=0.95, bottom=0.05
        )
        ax = fig.add_subplot(gs[1, :])
        c_star = 0
        if eps is not None:
            c_star = df[metric].min() - eps
        lines = []
        for solver_name in to_plot:
            df_solver = df.query("solver_name == @solver_name")
            solver = df_solver.iloc[0, -1]
            style = STYLES['*'].copy()
            style.update(STYLES[solver])
            curve = (
                df_solver.groupby('stop_val').quantile([0.2, 0.5, 0.8])
                .unstack()
            )
            lines.append(ax.semilogy(
                curve[('time', 0.5)], curve[(metric, 0.5)] - c_star,
                **style
            )[0])

            print("Min score:", df[metric].min())
            # lines.append(ax.semilogy(
            #     med_curve['time'], med_curve[metric],  # - c_star,
            #     **style
            # )[0])
            # med_idx = mode(
            #     df_solver.groupby('stop_val')[['time', 'idx_rep']]
            #     .apply(
            #         lambda x: None if len(x) <= 5 else
            #         x.sort_values('time').iloc[4]['idx_rep']
            #     )
            # ).mode
            # med_curve = df_solver.query("idx_rep == @med_idx")
            # plt.fill_betweenx(
            #     curve[(metric, 0.5)] - c_star,
            #     curve[('time', 0.2)], curve[('time', 0.8)],
            #     color=style['color'], alpha=0.3
            # )
            plt.fill_between(
                curve[('time', 0.5)],
                curve[(metric, 0.2)] - c_star,
                curve[(metric, 0.8)] - c_star,
                color=style['color'], alpha=0.3
            )
        print("Min score:", df[metric].min())
        ax.set_xlabel('Time [sec]')
        ax.set_xlim(xlim)

        ax_legend = fig.add_subplot(gs[0, :])
        ax_legend.set_axis_off()
        ax_legend.legend(handles=lines, ncol=3, loc='center')

        fig.savefig(fname.with_suffix('.pdf'))
        plt.show()
