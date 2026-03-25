# Global Convergence of the Canonical Evolutionary Strategy

This repository contains the official implementation and numerical experiments for the paper:
**"Global Convergence of the Canonical Evolutionary Strategy: From Mean-Field Limits to Semiclassical Concentration"** (2026).

## 📋 Overview
This project implements the **Canonical Evolutionary Strategy (CES)**, a global optimization framework. We provide the tools to reproduce the theoretical validation and the benchmark results presented in the paper.

### Key Features:
* **CES Engine**: Core implementation of the evolutionary dynamics.
* **Mean-Field Limits**: Numerical verification of the $M^{-1/2}$ convergence rate.
* **Semiclassical Analysis**: Scripts for visualizing the concentration of the ground state.
* **High-Dimensional Benchmarks**: Scalability tests in $d=1$ to $d=30$ for Ackley function at different initialization regimes (Uniform and Shifted).

## 📁 Repository Structure
* `/ces_engine.py`: Main algorithm logic.
* `/plot_theory_figures.py`: Mathematical validation plots.
* `/plot_mass_transport.py`: 1D and 2D mass transport visualizations.
* `/run_benchmarks_pro.py`: High-performance simulation suite.
* `/generate_paper_table.py`: Pipeline for LaTeX table generation.
* `/data/`: (Stored in Parquet) Raw simulation results.

## 🛠 Installation & Usage
1. Clone the repo:
   `git clone https://github.com/inria-chile/ces-global-convergence.git`
2. Install dependencies:
   `pip install numpy pandas matplotlib scipy pyarrow`
3. Run benchmarks:
   `python run_benchmarks.py`

## 🎓 Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{ces2026global,
  title={Global Convergence of the Canonical Evolutionary Strategy: From Mean-Field Limits to Semiclassical Concentration},
  author={Neto, Mat{\'\i}as and Garay, Nicolas and Mart{\'\i}, Luis and Sanchez-Pi, Nayat},
  journal={arXiv preprint / PPSN 2026},
  year={2026},
}
