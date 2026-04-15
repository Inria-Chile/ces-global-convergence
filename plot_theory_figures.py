import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
from ces_engine import run_ces_1d, sol_rep_mutador, error, run_semi_class, master_seed, rng
from utils import sigma_1, sigma_2, sigma_3, sigma_1_argmax, sigma_2_argmax, sigma_3_argmax

# --- Reproductibility configuration

R=30
ss = np.random.SeedSequence(master_seed)
child_seeds = ss.spawn(R)

# --- Validation Parameters (Section 3.1 & 4.1) ---
a = 40.0         # Selection intensity
c = 4e-2         # Mutation coefficient (diffusion)
x_min = -2.0     # Domain lower bound
x_max = 2.0      # Domain upper bound
u0 = lambda x: 0.25*np.ones_like(x) # Initial uniform density 
T = 3.0          # Final simulation time
M = 200_000      # Population size for high-fidelity mean-field validation
alpha = 0.5      # Power-law scaling factor [cite: 59]
x0 = rng.uniform(-2.0, 2.0, size=M) #Fixed initial condition

# --- Plot Functions

def plot_mean_field_validation(xx, tt, U, final_state, sigma_fun, sigma_argmax):

    plt.rcParams.update({
        "text.usetex": False,     
        "mathtext.fontset": "stix", 
        "font.family": "STIXGeneral",
        "font.size": 12
    })

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.hist(final_state, bins=100, range=(-2.0, 2.0), density=True, 
             color='skyblue', alpha=0.7, 
             label=rf"CES Population $\nu^M(t={tt[-1]:.1f})$")

    ax1.plot(xx, U[-1], color='navy', lw=2.5, 
             label=rf"PDE Solution $u(t={tt[-1]:.1f}, x)$")

    ax1.set_xlabel(r"State $x$")
    ax1.set_ylabel(r"Probability Density")
    
    ax1.set_ylim(0, np.max(U[-1]) * 1.3) 

    ax2 = ax1.twinx()
    
    ax2.fill_between(xx, sigma_fun(xx), color='gray', alpha=0.18, 
                     label=r"Fitness $\sigma_1(x)$")
    
    ax2.set_ylabel(r"Fitness Value $\sigma_1$", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, np.max(sigma_fun(xx)) * 1.3) 

    if not isinstance(sigma_argmax, (list, np.ndarray)):
        argmax_list = [sigma_argmax]
    else:
        argmax_list = sigma_argmax

    for k, pos in enumerate(argmax_list):
        label_m = "Global Maxima" if k == 0 else None 
        ax1.axvline(pos, lw=1.5, color="#d62728", ls="--", label=label_m, zorder=6)

    plt.title("CES Mass Transport vs. Replicator-Mutator PDE")
    ax1.grid(True, linestyle='--', alpha=0.4)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True)

    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    save_path = "figures/sigma_1_validation.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    
    #plt.show()
def save_data_convergence(M_list,tt,xx,U):

    results_list = []

    for i in range(R):
        rng_seed = np.random.default_rng(child_seeds[i])
        for j, M_val in enumerate(M_list):
            x0_M = rng_seed.uniform(x_min, x_max, size=M_val)
            
            pop_history = run_ces_1d(tt, x0_M, sigma_1, a, c, alpha)
            
            err_val = error(pop_history, U, tt, xx)
            
            results_list.append({
                "seed": i,
                "M": M_val,
                "error_W1": err_val
            })
            
        print(f"Progress: Seed {i+1}/{R} completed.")    

    df_results = pd.DataFrame(results_list)

    if not os.path.exists("data"):
        os.makedirs("data")
    
    df_results.to_parquet("data/convergence_results.parquet")

    return df_results

def plot_convergence_error(df, R):

    stats = df.groupby("M")["error_W1"].agg(["mean", "std"]).reset_index()
    
    M_val = stats["M"].values
    mean_err = stats["mean"].values
    std_err = stats["std"].values

    plt.figure(figsize=(8, 5))
    
    plt.fill_between(M_val, mean_err - std_err, mean_err + std_err, 
                     color="#1f77b4", alpha=0.2, label=r"$\pm 1$ Std. dev.")
    
    plt.plot(M_val, mean_err, 'o-', color="#1f77b4", lw=2, markersize=4, label="Mean $W_1$ error")

    theoretical = mean_err[0] * (M_val[0]/M_val)**0.5
    plt.plot(M_val, theoretical, 'k--', alpha=0.5, label=r"$\mathcal{O}(M^{-1/2})$")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Population size ($M$)")
    plt.ylabel(r"Error ($\overline{W}_1$)")
    plt.title(f"Mean-Field Limit Convergence ($R={R}$ seeds)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    
    plt.savefig("figures/Error_mean_field_convergence.pdf", bbox_inches="tight")
    plt.show()

def eigenfunction_concentration(c_list,sigma):
    x_list = []
    eigen_vals_list = []
    eigen_vecs_list = []
    for c in c_list:
        x, eigen_vecs, eigen_vals = run_semi_class(a, c, sigma, nx=400, xmin=x_min, xmax=x_max, neigen=1, normalize=True)
        eigen_vecs = np.abs(eigen_vecs.flatten())
        dx = x[1]-x[0]
        norm = -eigen_vals[0]/ np.trapezoid(a*sigma(x)*np.abs(eigen_vecs), x, dx) 
        x_list.append(x)
        eigen_vals_list.append(eigen_vals)
        eigen_vecs_list.append(eigen_vecs)
    return x_list, eigen_vals_list, eigen_vecs_list

def plot_eigenfunction_concentration(x_list, eigen_vecs_list, c_list, sigma_fun, sigma_argmax):
    plt.rcParams.update({
        "text.usetex": False,     
        "mathtext.fontset": "stix", 
        "font.family": "STIXGeneral",
        "font.size": 12
    })

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    xx = np.linspace(np.min(x_list[0]), np.max(x_list[0]), 500)
    yy = sigma_fun(xx)
    ax2.fill_between(xx, yy, alpha=0.18, color='gray', label=r"Fitness $\sigma(x)$ shape")
    
    color_axis_sec = (0.3, 0.3, 0.3, 0.7)
    ax2.set_ylabel(r"Fitness Landscape $\sigma(x)$", color=color_axis_sec, fontsize=11)
    ax2.tick_params(axis='y', colors=color_axis_sec)
    ax2.spines['right'].set_color(color_axis_sec)
    ax2.set_ylim(0, np.max(yy) * 1.3)

    if isinstance(c_list, (list, np.ndarray)) and len(c_list) > 1:
        indices = [n for n in range(len(c_list)) if n % 10 == 0 and n >= 50]
        if not indices:
            indices = [len(c_list) - 1]
        colors = cm.viridis(np.linspace(0, 0.85, len(indices)))
        for i, n in enumerate(indices):
            current_x = x_list[n]
            current_phi = eigen_vecs_list[n]
            
            label_c = rf"$c={c_list[n]:.3f}$" if len(c_list) > 1 else r"Principal Eigenfunction $\phi_1$"
            
            ax1.plot(current_x, current_phi, 
                     label=label_c, 
                     color=colors[i], 
                     lw=2.2, zorder=5)
    else:
        indices = [0]
        ax1.plot(x_list[0], eigen_vecs_list[0], color='#1f77b4', lw=3, label=r"$\phi_1(x)$ ($c=0.04$)", zorder=5)

    if not isinstance(sigma_argmax, (list, np.ndarray)):
        argmax_list = [sigma_argmax]
    else:
        argmax_list = sigma_argmax

    for k, pos in enumerate(argmax_list):
        label_m = "Global Maximum" if k == 0 else None
        ax1.axvline(pos, lw=1.5, color="#d62728", ls="--", label=label_m, zorder=6)

    ax1.set_xlabel(r"$x$", fontsize=12)
    ax1.set_ylabel(r"Principal Eigenfunction $\phi_1(x)$", fontsize=12)
    if len(c_list)>1:
        ax1.set_title(r"Concentration of $\phi_1$ for increasing $c$", fontsize=14, pad=20)
    else:
        ax1.set_title(f"Concentration of $\phi_1$ for fixed $c={c_list[0]}$", fontsize=14, pad=20)
    ax1.grid(True, ls=":", alpha=0.3)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=12, frameon=True, shadow=False)

    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    plt.tight_layout()
    max_num=len(argmax_list)
    suffix = f"multi_{max_num}" if len(argmax_list) > 1 else "single"
    plt.savefig(f"figures/eigenvec_concentration_{suffix}.pdf", bbox_inches="tight")
    #plt.show()
def plot_eigenvalue_stability(c_vals, eigen_vals_list):
    plt.figure(figsize=(7, 4))
    plt.plot(c_vals, eigen_vals_list, 'o-', color='darkgreen', markersize=3)
    plt.axhline(0, color='red', linestyle='--')
    plt.xscale('log')
    plt.xlabel(r"Mutation coefficient $c$")
    plt.ylabel(r"Principal Eigenvalue $\lambda_1$")
    plt.title("Spectral Stability Validation")
    plt.grid(True, alpha=0.3)
    #plt.show()

# --- Execution Block ---
if __name__ == "__main__":

    # --- Execute Figure 1(a): Mean-Field Validation ---

    x0 = rng.uniform(x_min, x_max, size=M)

 
    tt, xx, U = sol_rep_mutador(
        a, sigma_1, u0, c, T,
        nx=400, xmin=x_min, xmax=x_max,
        dt=0.001, renormalize=False
    )

    final_state = run_ces_1d(T, x0, sigma_1, a, c, alpha)

    plot_mean_field_validation(xx, tt, U, final_state, sigma_1, sigma_1_argmax)

    print("Figure 1a Finish")

    # --- Execute Figure 1(b): Convergence Analysis ---
    
    M_list = np.unique(np.logspace(2, 5, 50).astype(int))

    df_results=save_data_convergence(M_list, tt, xx, U)

    # df_results= pd.read_parquet("~/data/convergence_results.parquet")
    plot_convergence_error(df_results, R)
    print("Figure 1b Finish")

    # --- Execute Figure 2, 3 and 4: Semiclassical Analysis ---

    c_list = 4*np.logspace(-8, 2, 100)

    x_list, eigen_vals_list, eigen_vecs_list = eigenfunction_concentration(c_list, sigma_1)
    
    plot_eigenfunction_concentration(x_list, eigen_vecs_list, c_list, sigma_1, sigma_1_argmax)
    print("Figure 2 Finish")
    # plot_eigenvalue_stability(c_list, eigen_vals_list) # Uncomment to verify lambda_1 < 0

    c_list=[0.04]
    x_list, eigen_vals_list, eigen_vecs_list = eigenfunction_concentration(c_list, sigma_2)
    plot_eigenfunction_concentration(x_list, eigen_vecs_list, c_list, sigma_2, sigma_2_argmax)
    print("Figure 3 Finish")

    x_list, eigen_vals_list, eigen_vecs_list = eigenfunction_concentration(c_list, sigma_3)
    plot_eigenfunction_concentration(x_list, eigen_vecs_list, c_list, sigma_3, sigma_3_argmax)
    print("Figure 4 Finish")
