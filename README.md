# MagNet
Predicting Tokamak guiding-center trajectories using PyTorch-based Symplectic Neural Networks (SympNet) trained on GORILLA simulation data.

**Dependencies:**
- PyTorch
- NumPy
- SciPy
- Matplotlib

---

## Project Status: Entering Phase 2 (Model Training)

### ✅ Phase 1: Data Generation (Complete)
- **High-Fidelity Ground Truth:** Successfully utilized the GORILLA (Guiding-center ORbit Integration with Local Linearization Approach) engine to generate baseline collisionless phase-space trajectories.
- **Optimized Pipeline:** Implemented a monolithic, memory-safe data pipeline that circumvents Fortran grid-allocation RAM spikes to natively batch and simulate 1,000 simultaneous particle orbits.
- **Canonical Flux Coordinates:** Extracted orbital data and manually isolated the canonical momentum variables $(P_\theta, P_\phi, \theta, \phi)$.
- **Mathematical Splining:** Post-processed the adaptive RK45 integrator outputs using cubic splines to resample the trajectories into perfectly uniform time-steps. Angle wraparound anomalies (Runge's phenomenon at $2\pi$) were successfully resolved using a `unwrap-spline-wrap` algorithm.
- **Symplectic Normalization:** Implemented Conformal Symplectic Scaling to reduce the phase space volume from $10^7$ down to $\mathcal{O}(1)$ without distorting the Hamiltonian geometry.

### ⏳ Phase 2: Network Architecture (In Progress)
- **Objective:** Construct and evaluate PyTorch-based Symplectic Neural Networks.
- **Current Step:** Transitioning from **LA-SympNet** to **G-SympNet**. We empirically proved that LA-SympNets are mathematically ill-equipped to handle low-dimensional, high-frequency physical potentials (like a Tokamak magnetic field) due to parameter scaling ($O(d^2)$) causing vanishing gradients. The G-SympNet uses a wide MLP to inject sufficient width (capacity) into the network.

---

### ⚠️ Note on GORILLA Modifications
In order to correctly train the SympNets, the neural network requires inputs to be perfectly *canonically conjugate* pairs. Because GORILLA natively outputs non-canonical tracking variables ($v_\parallel$ and $s$), we had to slightly modify its Fortran source code to explicitly compute the true canonical momenta.

Specifically, `GORILLA/SRC/supporting_functions_mod.f90` was modified to include `p_theta_func`, and `gorilla_plot_mod.f90` was modified so that the trajectory output file yields 7 columns instead of 5: `[t, s, theta, phi, vpar, P_theta, P_phi]`. If you clone a fresh version of GORILLA, you will need to re-apply these Fortran hacks to extract the canonical momenta.
