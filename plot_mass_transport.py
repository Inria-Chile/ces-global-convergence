import numpy as np
import matplotlib.pyplot as plt
import os
from ces_engine import rng, run_ces_rd, cbo, sgd_ackley
from utils import ackley

# Hyperparameters ES
a = 40.0
c = 4e-2
M = 20_000
alpha = 0.5
Tf = 1.0

# Hyperparameters SGD
lr = 1e-2
n_steps_SGD = 20_000

# Hyperparameters CBO
n_steps_CBO = int(Tf*M**alpha)
dt = Tf/n_steps_CBO
lam = a
sigma_CBO = c
beta =  40.0

def simulation(d,mu=2.0,sigma_init=0.5):
    x_opt = np.zeros(d)
    x0 = mu+sigma_init * rng.standard_normal(size=(M, d))

    final_state = run_ces_rd(Tf, x0,  ackley, a, c, alpha)
    ack = ackley(final_state).reshape(-1)
    idx = np.argmin(ack)
    sol_EE = final_state[idx]
    sol_CBO = cbo(ackley, x0, n_steps_CBO, dt, lam, sigma_CBO, beta)
    sol_CBO_ = sol_CBO[1]
    pob_CBO = sol_CBO[0]

    #SGD
    x0_sgd = np.mean(x0, axis=0) 
    sol_SGD = sgd_ackley(x0_sgd, lr, n_steps_SGD) 
    return final_state, pob_CBO, sol_SGD
# ---  Plot functions

def plot_ackley_1d_validation(final_state, pob_CBO, sol_SGD):
    """
    Generates the 1D Ackley validation plot (Figure 1).
    
    """
    def ackley_1d(x):
        return -20 * np.exp(-0.2 * np.abs(x)) - \
               np.exp(np.cos(2 * np.pi * x)) + 20 + np.e

    plt.figure(figsize=(8, 7)) 
    plt.clf()

    x_plot = np.linspace(-3, 3, 1000) 
    y_plot = ackley_1d(x_plot)
    plt.plot(x_plot, y_plot, color='gray', alpha=0.3, lw=1.5, label="Ackley Landscape", zorder=1)
    plt.fill_between(x_plot, y_plot, alpha=0.05, color='gray')

    plt.scatter(final_state.flatten(), ackley_1d(final_state.flatten()), 
                color="#1f77b4", s=5, alpha=0.3, label="CES Population", 
                edgecolor='none', zorder=2)

    plt.scatter(pob_CBO.flatten(), ackley_1d(pob_CBO.flatten()), 
                color="#d62728", s=5, alpha=1.0, label="CBO Consensus", 
                edgecolor='none', zorder=3)

    plt.scatter(sol_SGD, ackley_1d(sol_SGD), 
                color="#27d630", s=40, marker='*', alpha=1, label="SGD Solution", 
                edgecolor='black', linewidth=0.5, zorder=4)


    plt.axvline(0, color='black', ls='--', lw=1, alpha=0.6, zorder=1)
    plt.scatter(0, 0, color='black', marker='+', s=50, lw=1.5, label=r"$x^*$", zorder=10)

    plt.xlim(-3, 3) 
    plt.ylim(-0.2, 12) 

    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

    plt.legend(loc='upper right', frameon=True, fontsize='small')
    plt.title("Final Solutions: 1D Ackley", pad=15)
    plt.xlabel("$x$")
    plt.ylabel("$A(x)$")

    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    save_path = "figures/Ackley_1d_dist.pdf"
    plt.savefig(save_path) 
    print(f"Figure saved successfully at: {save_path}")
    
    # plt.show()
def plot_ackley_2d_validation(final_state, pob_CBO, sol_SGD):
    """
    Generates the 2D Ackley validation plot (Figure 2).
    
    Uses rasterization for the complex contour background to maintain 
    performance while keeping scatter points as high-quality vectors.
    """
    def ackley_2d(x, y):
        term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        return term1 + term2 + 20 + np.e

    plt.figure(figsize=(8, 7))
    plt.clf()

    x_range = np.linspace(-3, 3, 500)
    y_range = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x_range, y_range)
    Z = ackley_2d(X, Y)

    plt.contourf(X, Y, Z, levels=100, cmap='Grays', alpha=0.4, zorder=1.5)

    plt.scatter(final_state[:, 0], final_state[:, 1], 
                color="#1f77b4", s=5, alpha=0.3, label="CES Population", zorder=2)
    
    plt.scatter(pob_CBO[:, 0], pob_CBO[:, 1], 
                color="#d62728", s=10, alpha=1.0, label="CBO Consensus", zorder=3)
    
    plt.scatter(sol_SGD[0], sol_SGD[1], 
                color="#27d630", s=100, marker='*', edgecolor='black', 
                label="SGD Solution", zorder=4)
    
    plt.scatter(0, 0, color='black', marker='+', s=20, lw=1.5, label=r"$x^*$", zorder=10)

    plt.gca().set_rasterization_zorder(2)

    plt.gca().set_aspect("equal")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    
    plt.legend(loc='lower right', frameon=True, fontsize='small')
    plt.title("Final Solutions: 2D Ackley", pad=15) 
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    save_path = "figures/Ackley_2d.pdf"
    plt.savefig(save_path, dpi=300) 
    print(f"Figure saved successfully at: {save_path}")
    
    # plt.show()
if __name__ == "__main__":
    
    # --- Execute code Figure 5

    final_state, pob_CBO, sol_SGD = simulation(d=1)

    plot_ackley_1d_validation(final_state,pob_CBO,sol_SGD)

    # Execute code Figure 6

    final_state, pob_CBO, sol_SGD = simulation(d=2)

    plot_ackley_2d_validation(final_state,pob_CBO,sol_SGD)