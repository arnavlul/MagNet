# MagNet
Predicting Tokamak guiding-center trajectories using PyTorch-based Symplectic Neural Networks (SympNet) trained on GORILLA simulation data.

## Project Status: 🚀 Entering Phase 2 (Model Training)

### ✅ Phase 1: Data Generation (Complete)
- **High-Fidelity Ground Truth:** Successfully utilized the GORILLA (Guiding-center ORbit Integration with Local Linearization Approach) engine to generate baseline collisionless phase-space trajectories.
- **Optimized Pipeline:** Implemented a monolithic, memory-safe data pipeline that circumvents Fortran grid-allocation RAM spikes to natively batch and simulate 1,000 simultaneous particle orbits.
- **Symmetry Flux Coordinates:** Extracted orbital data natively in $(s, \vartheta, \varphi)$ flux coordinates alongside pitch parameter $\lambda$.
- **Uniform Time-Step Interpolation:** Post-processed the adaptive RK45 integrator outputs using cubic splines to resample the trajectories into perfectly uniform time-steps, fully preparing the dataset for neural network training.

### ⏳ Phase 2: Network Architecture (In Progress)
- **Objective:** Construct and evaluate PyTorch-based Symplectic Neural Networks.
- **Current Step:** Preparing to build and benchmark the **LA-SympNet** (Linear-Activation) and **G-SympNet** (Gradient-based) architectures.


