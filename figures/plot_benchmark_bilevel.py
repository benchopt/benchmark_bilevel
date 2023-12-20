from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse

mpl.rc('text', usetex=True)

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
    '*': dict(lw=1.5),

    'amigo': dict(color='#5778a4', label=r'AmIGO'),
    'mrbo': dict(color='#e49444', label=r'MRBO'),
    'vrbo': dict(color='#e7ca60', label=r'VRBO'),
    'saba': dict(color='#d1615d', label=r'SABA'),
    'stocbio': dict(color='#85b6b2', label=r'StocBiO'),
    'srba': dict(color='#6a9f58', label=r'\textbf{SRBA}', lw=2),
    'pzobo': dict(color='#17becf', label=r'PZOBO'),
    'optuna': dict(color='#bcbd22', label=r'Optuna'),
}

N_CALLS = {
    # One loop
    'mrbo': (24, 4),  # inner, outer
    'sustain': (24, 4),
    'ttsa': (11, 2),
    'fsla': (4, 3),

    # Two loops solvers
    'amigo': (21, 2),
    'stocbio': (21, 2),
    'bsa': (21, 2),

    # Our solves
    'saba': (3, 2),
    'bio-svrg': (3, 2),
    'srba': (3, 2),
}

LEGEND_OUTSIDE = False

DEFAULT_WIDTH = 3.25
DEFAULT_DOUBLE_WIDTH = 6.75
DEFAULT_HEIGHT = 2.


def get_param(name, param='period_frac'):
    params = {}
    for vals in name.split("[", maxsplit=1)[1][:-1].split(","):
        k, v = vals.split("=")
        if v.replace(".", "").isnumeric():
            params[k] = float(v)
        else:
            params[k] = v
    return params[param]


def drop_param(name, param='period_frac'):
    new_name = name.split("[", maxsplit=1)[0] + '['
    for vals in name.split("[", maxsplit=1)[1][:-1].split(","):
        k, v = vals.split("=")
        if k != param:
            new_name += f'{k}={v},'
    return new_name[:-1] + ']'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot benchmarks results for bilevel optimization.'
    )
    parser.add_argument('--n-points', '-n', type=int, default=500,
                        help='# of points in the grid for interpolation.')
    parser.add_argument('--x-axis', '-x', type=str, default='time',
                        choices=['time', 'calls'],
                        help='Plot in time or number of calls to oracles.')
    parser.add_argument('--benchmark', '-b', type=str, default='ijcnn1',
                        choices=['ijcnn1', 'datacleaning0_5',
                                 'datacleaning0_7', 'datacleaning0_9',
                                 'covtype'],
                        help='Choose the benchmark to plot.')
    parser.add_argument('--criterion', '-c', type=str, default='100',
                        choices=['100', 'all'],
                        help='Choose the best curves with respect to the 100 \
                        first iterates or all the iterates.')
    args = parser.parse_args()

    x_axis = args.x_axis  # 'calls' or 'time'

    n_points = args.n_points

    bench = args.benchmark

    BENCHMARKS_CONFIG = dict(
        ijcnn1=(
            "ijcnn1.parquet", 'objective_value_func',
            'objective_value', ((1, 480), (0, 2e9)), 1e-10,
            r'$\|\nabla h(x^t)\|^2$', 'log', ('linear', 'linear'), None,
            64, 2**17, 49_990, 91_701
        ),
        datacleaning0_5=(
            "datacleaning0_5.parquet",
            'objective_value', 'objective_test_accuracy',
            ((.1, 900), (2e4, 5e7)), None, 'Test error', 'log',
            ('log', 'log'), (None, 40), 64, 2**5, 20_000, 5_000
        ),
        datacleaning0_7=(
            "datacleaning0_7.parquet",
            'objective_value', 'objective_test_accuracy',
            ((.1, 120), (8e3, 4e7)), None, 'Test error', 'log',
            ('log', 'log'), (None, 40), 64, 2**5, 20_000, 5_000
        ),
        datacleaning0_9=(
            "datacleaning0_9.parquet",
            'objective_value', 'objective_test_accuracy',
            ((.1, 1000), (2e4, 4e7)), None, 'Test error', 'log',
            ('log', 'log'), (None, 40), 64, 2**5, 20_000, 5_000
        ),
        covtype=(
            "covtype.parquet",
            'objective_value', 'objective_test_accuracy',
            ((.1, 1200), (5e4, 1e8)), None, 'Test error', 'log',
            ('log', 'log'), (27, 45), 512, 2**5, 371_847, 92_962
        ),
    )

    fname, metric_selection, metric_plot, xlim, eps, yname, yscaling, \
        xscaling, ylim, batch_size, eval_freq, n_inner_samples, \
        n_outer_samples = BENCHMARKS_CONFIG[bench]
    xlim = xlim[0] if x_axis == 'time' else xlim[1]
    xscaling = xscaling[0] if x_axis == 'time' else xscaling[1]

    fname = FILE_NAME.parent / fname
    print(fname)

    df = pd.read_parquet(fname)

    # normalize names
    df['solver'] = df['solver_name'].apply(
        lambda x: x.split('[')[0].lower()
    )
    df['seed'] = df['solver_name'].apply(
        lambda x: get_param(x, 'random_state')
    )

    df['solver_name'] = df['solver_name'].apply(
        lambda x: drop_param(x, 'random_state')
    )

    # keep only runs all the random seeds
    df['full'] = False
    n_seeds = df.groupby('solver_name')['seed'].nunique()
    for s in n_seeds.index:
        if n_seeds[s] == 10:
            df.loc[df['solver_name'] == s, 'full'] = True
    df = df.query('full == True')

    df.loc[
        df['solver_name'].apply(lambda x: 'full' in x), 'solver'
    ] = 'soba full batch'

    # Select curve that reach the lowest point
    if args.criterion == '100':
        to_plot = (
            df.query('stop_val <= 100')
            .groupby(['solver', 'solver_name', 'stop_val'])
            .median(metric_selection)
            .reset_index().sort_values(metric_selection)
            .groupby('solver').first()['solver_name']
        )
    elif args.criterion == 'all':
        to_plot = (
            df
            .groupby(['solver', 'solver_name', 'stop_val'])
            .median(metric_selection)
            .reset_index().sort_values(metric_selection)
            .groupby('solver').first()['solver_name']
        )
    to_plot = [to_plot[p] for p in STYLES if p in to_plot]
    df = df.query("solver_name in @to_plot")
    df.to_parquet(f'{fname.stem}_best_param.parquet')

    solvers = [s for s in STYLES if s in df['solver'].values]
    print(solvers)
    to_plot = df.set_index('solver').loc[solvers, 'solver_name']
    to_plot = to_plot.unique()

    print("Chosen parameters:")
    for s in to_plot:
        print(f"- {s}")

    legend_ratio = 0.1
    fig = plt.figure(
        figsize=(DEFAULT_WIDTH, DEFAULT_HEIGHT * (1 + legend_ratio))
    )
    if LEGEND_OUTSIDE:
        legendFig = plt.figure()
    gs = plt.GridSpec(
        2, 1, height_ratios=[legend_ratio, 1], top=0.95, bottom=0.05
    )
    ax = fig.add_subplot(gs[1, :])
    c_star = 0
    if eps is not None:
        c_star = (
            df.groupby(['solver', 'stop_val'])
            .median(numeric_only=True).loc[:, metric_plot].min() - eps
        )

    # if metric == 'objective_value':
    #     axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    lines = []
    for solver_name in to_plot:
        df_solver = df.query("solver_name == @solver_name")
        solver = df_solver.iloc[0]['solver']
        style = STYLES['*'].copy()
        style.update(STYLES[solver])
        curves = [data[['time', metric_plot]].values
                  for _, data in df_solver.groupby('seed')]
        vals = [c[:, 1] for c in curves]
        if x_axis == 'time':
            times = [c[:, 0] for c in curves]
            tmin = np.min([np.min(t) for t in times])
            tmax = np.max([np.max(t) for t in times])
            # time_grid = np.geomspace(tmin, xlim[1] + 1, n_points)
            time_grid = np.linspace(np.log(tmin), np.log(xlim[1] + 1),
                                    n_points)
            interp_vals = np.zeros((len(times), n_points))
            # import ipdb; ipdb.set_trace()
            for i, (t, val) in enumerate(zip(times, vals)):
                interp_vals[i] = np.exp(np.interp(time_grid, np.log(t),
                                        np.log(val)))
            if metric_plot == 'objective_test_accuracy':
                interp_vals *= 100
            time_grid = np.exp(time_grid)
            medval = np.quantile(interp_vals, .5, axis=0)
            q1 = np.quantile(interp_vals, .2, axis=0)
            q2 = np.quantile(interp_vals, .8, axis=0)
            curve = (
                df_solver.groupby('stop_val').quantile([0.2, 0.5, 0.8],
                                                       numeric_only=True)
                .unstack()
            )
            lines.append(ax.semilogy(
                time_grid, medval - c_star,
                **style
            )[0])
            ax.fill_between(
                time_grid,
                q1 - c_star,
                q2 - c_star,
                color=style['color'], alpha=0.3
            )
        elif x_axis == 'calls':
            n_inner_calls, n_outer_calls = N_CALLS[solver]
            if 'full' in solver:
                n_inner_calls *= n_inner_samples
                n_outer_calls *= n_outer_samples
            else:
                n_inner_calls *= batch_size
                n_outer_calls *= batch_size
            calls = [
                np.arange(1, c.shape[0] + 1) *
                (n_inner_calls + n_outer_calls) * eval_freq
                for c in curves
            ]
            # We first translate the calls grid to the right to avoid
            # calls[i][0] = 0 in the logarithmic interpolation
            nmin = np.min([np.min(n) for n in calls])
            nmax = np.max([np.max(n) for n in calls])
            calls_grid = np.linspace(np.log(nmin),
                                     np.log(xlim[1] +
                                     (n_inner_calls + n_outer_calls)
                                     * 2**17),
                                     n_points)
            interp_vals = np.zeros((len(calls), n_points))
            for i, (t, val) in enumerate(zip(calls, vals)):
                interp_vals[i] = np.exp(np.interp(calls_grid, np.log(t),
                                        np.log(val)))
            if metric_plot == 'objective_test_accuracy':
                interp_vals *= 100
            calls_grid = np.exp(calls_grid)
            # We shift the grid to the left for the plot
            calls_grid -= (n_inner_calls + n_outer_calls) * eval_freq

            medval = np.quantile(interp_vals, .5, axis=0)
            q1 = np.quantile(interp_vals, .2, axis=0)
            q2 = np.quantile(interp_vals, .8, axis=0)
            curve = (
                df_solver.groupby('stop_val').quantile([0.2, 0.5, 0.8],
                                                       numeric_only=True)
                .unstack()
            )

            lines.append(ax.semilogy(
                calls_grid,
                medval - c_star,
                **style
            )[0])
            ax.fill_between(
                calls_grid - 1,
                q1 - c_star,
                q2 - c_star,
                color=style['color'], alpha=0.3
            )

        print(f"Min score ({solver}):", df_solver[metric_selection].min())

    print("Min score:", df[metric_selection].min())
    if x_axis == 'time':
        x_ = ax.set_xlabel('Time [sec]')
    elif x_axis == 'calls':
        x_ = ax.set_xlabel('Number of calls to oracles')
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    y_ = ax.set_ylabel(yname)
    ax.grid()
    ax.set_yscale(yscaling)
    ax.set_xscale(xscaling)
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.set_axis_off()
    if LEGEND_OUTSIDE:
        legendFig.legend(handles=lines, ncol=2, loc='center')
        legendFig.savefig('legend.pdf')
        l_ = x_
    else:
        l_ = ax.legend(handles=lines, ncol=2, prop={'size': 6.5})
    if "datacleaning" in fname.stem:
        ticklist = [15, 20, 30, 40]
        labels = [r'$%d \%%$' % tick for tick in ticklist]
        # labels[-2] = ''
        ax.set_yticks(ticklist, labels=labels)
    elif "covtype" in fname.stem:
        ticklist = [30, 35, 40]
        labels = [r'$%d \%%$' % tick for tick in ticklist]
        # labels[-2] = ''
        ax.set_yticks(ticklist, labels=labels)

    # if metric == 'objective_value':
    #     x1, x2, y1, y2 = t_lim, xlim[1], .12, .17
    #     axins.set_xlim(x1, x2)
    #     axins.set_ylim(y1, y2)
    #     axins.set_xticklabels([])
    #     axins.set_yticklabels([])
    #     ax.indicate_inset_zoom(axins, edgecolor="black")
    if x_axis == 'time':
        fig.savefig(
            fname.with_suffix('.pdf'), bbox_extra_artists=[x_, y_, l_],
            bbox_inches='tight'
        )
    elif x_axis == 'calls':
        fig.savefig(
            Path(fname.stem + '_calls').with_suffix('.pdf'),
            bbox_extra_artists=[x_, y_, l_],
            bbox_inches='tight'
        )
    plt.close('all')
