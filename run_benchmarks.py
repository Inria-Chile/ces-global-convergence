import numpy as np
import pandas as pd
import os
from ces_engine import rng, run_ces_rd, cbo, sgd_ackley, master_seed
from utils import ackley

# Reproductibility

ss = np.random.SeedSequence(master_seed+1000)



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

def run_benchmarks_production(R=30, dimensions=[1, 2, 10, 30], case="uniform", threshold=1e-2):
    child_seeds = ss.spawn(R)
    results = []
    
    for d in dimensions:
        print(f"\n>>> Dimension d={d} | Case: {case} <<<")
        target = np.zeros(d)
        
        for i in range(R):
            rng_seed = np.random.default_rng(child_seeds[i])
            if case == "uniform":
                x0 = rng.uniform(-2.0, 2.0, size=(M, d))
            else:
                x0 = 2.0 + 0.5 * rng.standard_normal(size=(M, d))
            
            t_grid = np.linspace(0, Tf, n_steps_CBO + 1)
            xt_all = run_ces_rd(t_grid, x0, ackley, a, c, alpha)
            final_pop_ces = xt_all[-1]
            best_idx_ces = np.argmin(ackley(final_pop_ces))
            err_ces = np.linalg.norm(final_pop_ces[best_idx_ces] - target)
            
            results.append({
                "Method": "CES (Ours)", "Dim": d, "Seed": i, "Case": case,
                "Error": err_ces, "Success": 1 if err_ces < threshold else 0
            })

            _, best_cbo, _ = cbo(ackley, x0, n_steps_CBO, dt, lam, sigma_CBO, beta)
            err_cbo = np.linalg.norm(best_cbo - target)
            
            results.append({
                "Method": "CBO", "Dim": d, "Seed": i, "Case": case,
                "Error": err_cbo, "Success": 1 if err_cbo < threshold else 0
            })

            if d <= 2:
                x0_sgd = np.mean(x0, axis=0) 
                sol_sgd = sgd_ackley(x0_sgd,lr, n_steps_SGD) 
                err_sgd = np.linalg.norm(sol_sgd - target)
                
                results.append({
                    "Method": "SGD", "Dim": d, "Seed": i, "Case": case,
                    "Error": err_sgd, "Success": 1 if err_sgd < threshold else 0
                })

    df = pd.DataFrame(results)
    output_name = f"data/benchmarks_final_{case}.parquet"
    df.to_parquet(output_name, engine='pyarrow')
    print(f"\nData saved in {output_name}")
    return df
def to_latex_scientific_compact(val, std):
    if pd.isna(val) or val is None: 
        return "--"
    
    def fmt_latex(x):
        if x == 0: 
            return "0"
        if 0.01 <= abs(x) < 100:
            return f"{x:.2f}"
            
        exponent = int(np.floor(np.log10(abs(x))))
        coeff = x / (10**exponent)
        
        return f"{coeff:.1f} \\times 10^{{{exponent}}}"

    return f"${fmt_latex(val)}~(\pm~{fmt_latex(std)})$"

def generate_full_paper_table(df_uni, df_shi, filename="tables/final_results.tex"):
    df_uni['Scenario'] = 'Uniform'
    df_shi['Scenario'] = 'Shifted'
    df = pd.concat([df_uni, df_shi])

    stats = df.groupby(['Scenario', 'Dim', 'Method']).agg({
        'Error': ['mean', 'std'],
        'Success': 'mean'
    }).reset_index()
    stats.columns = ['Scenario', 'Dim', 'Method', 'Mean', 'Std', 'SR']
    pivot = stats.pivot_table(index=['Scenario', 'Dim'], columns='Method', values=['Mean', 'Std', 'SR'])
    pivot = pivot.reindex(['Uniform', 'Shifted'], level=0)
    
    methods = ['CES (Ours)', 'CBO', 'SGD']
    
    latex = []
    latex.append(r"\begin{table}[tb]")
    latex.append(r"\centering")
    latex.append(r"\caption{Benchmark Performance: Mean $L_2$ Error $\pm$ SD and Success Rate (SR).}")
    latex.append(r"\label{tab:final_results}")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    latex.append(r"\begin{tabular}{ll cc cc cc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Scenario} & \textbf{Dim} & \multicolumn{2}{c}{\textbf{CES (Ours)}} & \multicolumn{2}{c}{\textbf{CBO}} & \multicolumn{2}{c}{\textbf{SGD}} \\")
    latex.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}")
    latex.append(r"& & \textit{Error ($\pm$ SD)} & \textit{SR} & \textit{Error ($\pm$ SD)} & \textit{SR} & \textit{Error ($\pm$ SD)} & \textit{SR} \\")
    latex.append(r"\midrule")

    for scenario in ['Uniform', 'Shifted']:
        dims = sorted(df[df['Scenario'] == scenario]['Dim'].unique())
        first_row_scenario = True
        
        for d in dims:
            row_str = ""
            if first_row_scenario:
                row_str += f"\\multirow{{{len(dims)}}}{{*}}{{\\textbf{{{scenario}}}}}"
                first_row_scenario = False
            
            row_str += f" & $d={d}$"
            
            row_means = {m: pivot.loc[(scenario, d), ('Mean', m)] for m in methods if not pd.isna(pivot.loc[(scenario, d), ('Mean', m)])}
            best_method = min(row_means, key=row_means.get) if row_means else None

            for m in methods:
                try:
                    m_mean = pivot.loc[(scenario, d), ('Mean', m)]
                    m_std = pivot.loc[(scenario, d), ('Std', m)]
                    m_sr = pivot.loc[(scenario, d), ('SR', m)]
                    
                    if pd.isna(m_mean):
                        row_str += " & -- & --"
                    else:
                        err_str = to_latex_scientific_compact(m_mean, m_std)
                        sr_str = f"{m_sr*100:.0f}\\%"
                        
                        if m == best_method:
                            err_str = f"\\textbf{{{err_str}}}"
                            sr_str = f"\\textbf{{{sr_str}}}"
                            
                        row_str += f" & {err_str} & {sr_str}"
                except KeyError:
                    row_str += " & -- & --"
            
            row_str += r" \\"
            latex.append(row_str)
        
        if scenario == 'Uniform':
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}%")
    latex.append(r"}")
    latex.append(r"\end{table}")

    final_latex_code = "\n".join(latex)

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        
    with open(filename, "w") as f:
        f.write(final_latex_code)
    
    return final_latex_code

if __name__ == "__main__":
    df_uniform = run_benchmarks_production(R=30, dimensions=[1, 2, 10, 30], case="uniform", threshold=1e-2)
    df_shifted = run_benchmarks_production(R=30, dimensions=[1, 2, 10, 30], case="shifted", threshold=1e-2)
    #df_uniform = pd.read_parquet("~/data/benchmarks_final_uniform.parquet")
    #df_shifted = pd.read_parquet("~/data/benchmarks_final_shifted.parquet")
    table_code=generate_full_paper_table(df_uniform, df_shifted, filename="tables/final_results.tex")

    #print(table_code)