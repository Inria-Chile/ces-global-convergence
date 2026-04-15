import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import ackley
from ces_engine import cbo, run_ces_rd, master_seed

# Reproductibility

ss = np.random.SeedSequence(master_seed+1000)

# Hyperparameters ES
a = 40.0
c = 4e-2
M = 20_000
alpha = 0.5
Tf = 2.0

# Hyperparameters CBO
n_steps_CBO = int(Tf*M**alpha)
dt = Tf/n_steps_CBO
lam = a
sigma_CBO = c
beta =  40.0

def save_convergence_data(d, R, case="uniform", filename=None):
    t_grid = np.linspace(0, Tf, n_steps_CBO + 1)
    all_data = []
    x_opt = np.zeros(d)
    mu = 2.0 if case == "shifted" else 0.0
    child_seeds = ss.spawn(R)
    print(f"Running simulation: d={d}, case={case}...")
    for i in range(R):
        rng = np.random.default_rng(child_seeds[i])
        x0 = (rng.uniform(-2, 2, (M, d)) if case == "uniform" 
              else mu + 0.5 * rng.standard_normal((M, d)))
        
        # --- CBO ---
        _, _, history_CBO = cbo(ackley, x0, n_steps_CBO, dt, lam, sigma_CBO, beta, save_every=1)
        for k, h in enumerate(history_CBO):
            all_data.append({"Method": "CBO", "Time": t_grid[k], "Error": np.linalg.norm(h['c'] - x_opt), "Seed": i})

        # --- CES ---
        xt_all = run_ces_rd(t_grid, x0, ackley, a, c, alpha)
        for k, t in enumerate(t_grid):
            f_vals = ackley(xt_all[k])
            best_x = xt_all[k][np.argmin(f_vals)]
            all_data.append({"Method": "CES (Ours)", "Time": t, "Error": np.linalg.norm(best_x - x_opt), "Seed": i})

    df = pd.DataFrame(all_data)
    fname = filename or f"data/history_convergence_d{d}_{case}.parquet"
    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_parquet(fname)
    print(f"Done! Data saved to {fname}")
    return df
    

def plot_from_history(df,d,case):

    stats = df.groupby(['Method', 'Time'])['Error'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(10, 6))
    colors = {'CES (Ours)': '#1f77b4', 'CBO': '#d62728'}
    
    for method in stats['Method'].unique():
        m_stats = stats[stats['Method'] == method]
        plt.plot(m_stats['Time'], m_stats['mean'], label=method, color=colors[method], lw=2)
        plt.fill_between(m_stats['Time'], 
                         m_stats['mean'] - m_stats['std'], 
                         m_stats['mean'] + m_stats['std'], 
                         color=colors[method], alpha=0.15)

    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title(f'Convergence Profile (d={d}, {case} Initialization)')
    plt.legend()
    plt.xlabel("Time $t$")
    plt.ylabel("$L_2$ Error")

    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig(f'figures/final_convergence_d{d}_{case}.pdf', dpi=300)
    #plt.show()


if __name__ == "__main__":
    df=save_convergence_data(d=30, R=30, case="shifted")
    #df=pd.read_parquet("~/data/history_convergence_d30_shifted.parquet")
    plot_from_history(df,d=30,case="Shifted")