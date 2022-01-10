import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("bench_figure_1.csv")

fontsize = 11
params = {
    'axes.labelsize': fontsize + 2,
    'font.size': fontsize + 2,
    'legend.fontsize': fontsize + 2,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'text.usetex': True
}
plt.rcParams.update(params)

begining_solver = 'single-loop[step_size=0.01,outer_ratio=50,batch_size=1,vr='
end_solver = ']'
solvers = df["solver_name"].unique()
solvers = [
    s.replace(begining_solver, '').replace(end_solver, '')
    for s in solvers
]
solver_legend = dict(
    none="SGD",
    saga_z=r'SAGA on $z$',
    saga="SAGA"
)

fig = plt.figure(figsize=(4, 3))
for s in solvers:
    to_plot = df.loc[
        df["solver_name"] == begining_solver+s+end_solver
    ]
    plt.semilogy(
        to_plot["stop_val"],
        to_plot["objective_value"],
        label=solver_legend[s],
        linewidth=2
    )

plt.xlim(right=4000)
plt.ylim(bottom=1e-17)
plt.xlabel("Iterations")
plt.ylabel(r"$\|\nabla h (x_t)\|^2$")
plt.legend()
fig.tight_layout()
fig.savefig("figure_1.pdf", dpi=300)
