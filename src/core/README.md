# Core Lung Simulation Engine (`src/core/`)

This directory contains the foundational numerical solvers, continuum mechanics formulations, airway fluid networks, and optimization routines powering the ARDS Lung Simulator.

## Architecture & Module Overview

```
src/core/
├── vcv_lung.py                      # Central FEniCS solver for Volume-Controlled Ventilation
├── AirwayManager.py                 # Airway tree geometry, branching fluid dynamics & resistances
├── modelfunctions.py                # Hyperelastic/poroelastic constitutive laws & kinematic tensors
├── supportfunctions.py              # FEniCS solver configs, boundary measures & export utilities
├── BOCalibration.py                 # Bayesian Optimization calibration pipeline (scikit-optimize)
├── Generalized_BC_Functions.py      # Unified ventilator waveform generator (flow, pressure, volume)
├── determine_times_in_vcv_signals.py# Cycle event timing detector for VCV (inspiration, pause, expiration)
├── determine_times_in_pcv_signals.py# Cycle event timing detector for PCV waveforms
├── rescale_flow.py                  # Flow signal rescaling & volume normalization utility
├── functions2.py                    # 3D planar and mesh geometric helper routines
└── BidimLib.py                      # 2D slicing, regional compartment integration & plotting library
```

---

## Key Modules

### 1. `vcv_lung.py`
The primary simulation entrypoint for volume-controlled mechanical ventilation:
* Implements finite-deformation poroelastic equations in **FEniCS** (`dolfin`).
* Solves coupled equations for solid displacement $\mathbf{u}$ and pore pressure $p$.
* Couples terminal airway boundary fluxes to 3D lung parenchyma subdomains via `AirwayManager`.
* Computes local Jacobian determinants $J = \det(\mathbf{F})$, regional aeration compartments (overaerated, normally aerated, poorly aerated, non-aerated), stress tensors, and global respiratory compliance/elastance.

### 2. `AirwayManager.py`
A comprehensive object-oriented fluid network representation of the central and conducting tracheobronchial tree:
* Parses 3D airway centerlines and surface segmentations.
* Computes Poiseuille/Pedley flow resistances for each branch generation based on diameter, length, and flow Reynolds number.
* Solves terminal flow splitting and regional pressure drops from tracheal inlet to distal terminal branches.

### 3. `modelfunctions.py`
Defines continuum mechanics formulations and material strain energy functions $\Psi$:
* Implements hyperelastic strain energy models (Birzle compressible porous model, exponential, neo-Hookean).
* Computes first and second Piola-Kirchhoff stress tensors ($\mathbf{P}$, $\mathbf{S}$) and consistent material algorithmic tangent tensors.
* Formulates permeability laws capturing strain-dependent tissue porosity.

### 4. `supportfunctions.py`
Shared numerical and FEniCS integration routines:
* Sets up finite element function spaces ($P_1$, $P_2$, vector/tensor spaces).
* Configures Newton-Raphson non-linear solver parameters with line-search algorithms.
* Manages subdomain facet measures (`ds`), boundary condition application, and time-stepping export routines.

### 5. `BOCalibration.py`
Bayesian Optimization parameter calibration engine:
* Integrates `scikit-optimize` (`skopt`) with Latin Hypercube Sampling (`qmc.LatinHypercube`) and Gaussian Process surrogate modeling.
* Calibrates tissue stiffness parameters ($c_1, c_2$), base permeability ($k_0$), and regional airway resistances against measured airway pressure curves.

### 6. Waveform & Signal Processing Utilities
* **`Generalized_BC_Functions.py`**: Constructs continuous, interpolated ventilator flow waveforms from clinical data or mathematical profiles.
* **`determine_times_in_vcv_signals.py`**: Identifies cycle transition points (start of inspiration, inspiratory pause, end-expiration) to drive simulation time-stepping.
* **`rescale_flow.py`**: Rescales experimental flow series to target tidal volumes while preserving physiological waveforms.
