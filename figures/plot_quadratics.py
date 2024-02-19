from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('text', usetex=True)

FILE_NAME = Path(__file__).with_suffix('')
METRIC = 'objective_value'

# DEFAULT_WIDTH = 3.25
DEFAULT_WIDTH = 3
DEFAULT_HEIGHT = 2
LEGEND_RATIO = 0.1

N_POINTS = 500
X_LIM = 250

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
    'f2sa': dict(color='#bcbd22', label=r'F2SA'),
}


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
    fname = "quadratic.parquet"
    fname = FILE_NAME.parent / fname

    if Path(f'{fname.stem}_stable.parquet').is_file():
        df = pd.read_parquet(f'{fname.stem}_stable.parquet')
        print(f'{fname.stem}_stable.parquet')
    else:
        df = pd.read_parquet(fname)
        print(fname)

        # normalize names
        df['solver'] = df['solver_name'].apply(
            lambda x: x.split('[')[0].lower()
        )
        df['seed_solver'] = df['solver_name'].apply(
            lambda x: get_param(x, 'random_state')
        )
        df['seed_data'] = df['data_name'].apply(
            lambda x: get_param(x, 'random_state')
        )

        df['solver_name'] = df['solver_name'].apply(
            lambda x: drop_param(x, 'random_state')
        )
        df['data_name'] = df['data_name'].apply(
            lambda x: drop_param(x, 'random_state')
        )
        df['cond'] = df['data_name'].apply(
            lambda x: get_param(x, 'L_inner_inner')/get_param(x, 'mu_inner')
        )
        df['n_inner'] = df['data_name'].apply(
            lambda x: get_param(x, 'n_samples_inner')
        )
        df['n_outer'] = df['data_name'].apply(
            lambda x: get_param(x, 'n_samples_outer')
        )
        df['n_tot'] = df['n_inner'] + df['n_outer']

        # keep only runs all the random seeds
        df['full'] = False
        n_seeds = df.groupby('solver_name')['seed_data'].nunique()
        n_seeds *= df.groupby('solver_name')['seed_solver'].nunique()
        for s in n_seeds.index:
            if n_seeds[s] == 10:
                df.loc[df['solver_name'] == s, 'full'] = True
        df = df.query('full == True')
        df.to_parquet(f'{fname.stem}_stable.parquet')

    fig = plt.figure(
        figsize=(DEFAULT_WIDTH, DEFAULT_HEIGHT * (1 + LEGEND_RATIO))
    )

    gs = plt.GridSpec(
        len(df['n_tot'].unique()), len(df['cond'].unique()),
        height_ratios=[1] * len(df['n_tot'].unique()),
        width_ratios=[1] * len(df['cond'].unique()),
        hspace=0.5, wspace=0.3
    )

    lines = []
    for i, n_tot in enumerate(df['n_tot'].unique()):
        for j, cond in enumerate(df['cond'].unique()):
            df_pb = df.query("cond == @cond & n_tot == @n_tot")
            print(f"Cond: {cond}, n: {df_pb['n_inner'].iloc[0]}, "
                  + f"m: {df_pb['n_outer'].iloc[0]}")
            to_plot = (
                df.query("cond == @cond & n_tot == @n_tot & stop_val <= 100")
                .groupby(['solver', 'solver_name', 'data_name', 'stop_val'])
                .median(METRIC)
                .reset_index().sort_values(METRIC)
                .groupby('solver').first()[['solver_name']]
            )
            (
                df.query("solver_name in @to_plot.values.ravel()")
                .to_parquet(f'{fname.stem}_best_params.parquet')
            )
            print("Chosen parameters:")
            for s in to_plot['solver_name']:
                print(f"- {s}")
            ax = fig.add_subplot(gs[i, j])
            for solver_name in to_plot['solver_name']:
                df_solver = df_pb.query("solver_name == @solver_name")
                solver = df_solver['solver'].iloc[0]
                style = STYLES['*'].copy()
                style.update(STYLES[solver])
                curves = [data[['time', METRIC]].values
                          for _, data in df_solver.groupby(['seed_data',
                                                            'seed_solver'])]
                vals = [c[:, 1] for c in curves]
                times = [c[:, 0] for c in curves]
                tmin = np.min([np.min(t) for t in times])
                tmax = np.max([np.max(t) for t in times])
                time_grid = np.linspace(np.log(tmin), np.log(tmax + 1),
                                        N_POINTS)
                interp_vals = np.zeros((len(times), N_POINTS))
                for k, (t, val) in enumerate(zip(times, vals)):
                    interp_vals[k] = np.exp(np.interp(time_grid, np.log(t),
                                            np.log(val)))
                time_grid = np.exp(time_grid)
                medval = np.quantile(interp_vals, .5, axis=0)
                q1 = np.quantile(interp_vals, .2, axis=0)
                q2 = np.quantile(interp_vals, .8, axis=0)
                if i == 0 and j == 0:
                    lines.append(ax.semilogy(
                        time_grid, np.sqrt(medval),
                        **style
                    )[0])
                else:
                    ax.semilogy(
                        time_grid, np.sqrt(medval),
                        **style
                    )
                ax.fill_between(
                    time_grid,
                    np.sqrt(q1),
                    np.sqrt(q2),
                    color=style['color'], alpha=0.3
                )
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(r'$\|\nabla h(x^t)\|$')
                print(f"Min score ({solver}):", df_solver[METRIC].min())
            ax.grid()
            ax.set_xlim([0, X_LIM])

            if i == 0 and j == 0:
                ax_legend = ax.legend(
                    handles=lines,
                    ncol=2,
                    prop={'size': 6.5}
                )
    print(f"Saving {fname.with_suffix('.pdf')}")
    fig.savefig(
        fname.with_suffix('.pdf'),
        bbox_inches='tight',
        bbox_extra_artists=[ax_legend]
    )
