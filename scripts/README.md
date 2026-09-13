# Simulation Scripts & Execution Entrypoints (`scripts/`)

This directory contains the primary command-line interfaces (CLI) used to run forward finite element lung simulations, execute Bayesian optimization calibrations, and perform multi-parameter sensitivity sweeps.

## Available Runners

### 1. `2025-12_universal-executer.py`
The primary production runner for volume-controlled ventilation simulations. It coordinates mesh loading, airway tree initialization, boundary condition waveform application, and FEniCS solver invocation across experimental pig cohorts:

```bash
# Example: Run a simulation for Pig 5 with medium-coarse mesh resolution
python scripts/2025-12_universal-executer.py -pig_id 5 -mesh_type medium-coarse
```

**Supported Options**:
* `-pig_id`: Animal subject ID (choices: `2`, `3`, `4`, `5`, `6`; default: `5`).
* `-mesh_type`: Mesh resolution (choices: `coarse`, `medium-coarse`, `medium`, `medium-fine`, `fine`; default: `medium-coarse`).
* Supports command-line specification of input/output directories, time steps, and export frequencies.

---

### 2. `calibrate.py`
The execution driver for parameter calibration:
* Wraps the `BOCalibration` engine in `src/core/`.
* Fits nonlinear tissue stiffness coefficients, base permeability, and regional airway resistances to match experimental airway pressure curves.
* Supports distributed evaluation across multiple CPU cores via multiprocessing.

```bash
# Example: Launch Bayesian optimization calibration
python scripts/calibrate.py
```

---

### 3. `2026-02_sensibility-executer.py`
Batch execution script designed for parameter sensitivity analyses:
* Sweeps mechanical parameters (matrix stiffness $c_1$, fiber stiffness $c_2$, base permeability $k_0$, and peripheral resistance factors).
* Exports resulting pressure-volume loops, peak pressures, driving pressures, and aeration compartment distributions.

```bash
# Example: Run parameter sensitivity batch
python scripts/2026-02_sensibility-executer.py
```

---

> [!NOTE]
> **Environment Note**: These scripts require the core simulator package on the Python search path. When running from the repository root, ensure `PYTHONPATH` includes `src/core` or add `sys.path.append("src/core")`.
