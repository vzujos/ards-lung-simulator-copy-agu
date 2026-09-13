# Source Directory (`src/`)

This directory contains all source code for the ARDS Lung Simulator, structured into modular functional subpackages:

## Directory Layout

* **[`core/`](core/)**: The primary numerical simulation engine, including:
  * Finite element Volume-Controlled Ventilation solver (`vcv_lung.py`)
  * Tracheobronchial airway network fluid model (`AirwayManager.py`)
  * Hyperelastic/poroelastic constitutive laws and strain energy models (`modelfunctions.py`)
  * Numerical helpers, function spaces, and solver settings (`supportfunctions.py`)
  * Bayesian Optimization calibration pipeline (`BOCalibration.py`)
  * Ventilator boundary condition generation and timing detection utilities.

* **[`meshing/`](meshing/)**: The sequential 3-step meshing pipeline transforming 3D lung segmentations into FEniCS-ready tetrahedral meshes (`1_initial_meshing.py`, `2_extrabc_info.py`, `3_final_meshing_wbc.py`, `evaluate_meshes.py`).

* **[`postprocessing/`](postprocessing/)**: Results extraction, VTU field projection to reference anatomy (`simulation-vtu-grouper.py`), biomechanical aggregation (`biomechanical-analysis.py`), and model validation against experimental data (`evaluate_simulations.py`).

* **[`experiment/`](experiment/)**: Exploratory scripts, candidate kinematic models, and image analysis tools retained for evaluation and active development.

* **[`legacy/`](legacy/)**: Preserved legacy solvers (e.g., PCV solver), historical Nelder-Mead optimization code, and zipped multi-language archives (MATLAB, C++/CGAL).

* **`trash/`**: *Temporary staging folder containing superseded code excerpts and redundant scripts, slated for manual deletion.*
