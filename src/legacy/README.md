# Legacy Code & Historical Archives (`src/legacy/`)

This directory preserves older prototypes, superseded numerical solvers, early Nelder-Mead optimization routines, and archived MATLAB/C++ scripts. While not part of the active production pipeline, these files contain valuable historical context and algorithm reference implementations for future development.

## Preserved Code & Archives

```
src/legacy/
├── pcv_lung.py                           # Legacy FEniCS solver for Pressure-Controlled Ventilation (PCV)
├── executer_pig6_aprv.py                 # Dedicated APRV ventilation execution script for Pig 6
├── run-calibration.py                    # Early calibration runner using Nelder-Mead simplex search
├── nelder-mead.py                        # Standalone Nelder-Mead simplex optimization algorithm
├── ROIAnalysis.py                        # Historical Python 2 version of the ROI analysis module
├── airways_at_matlab.m                   # Legacy MATLAB airway geometry extraction script
├── __legacy__cgal-airway-meshing.zip     # Archived C++/CGAL airway skeletonization sample & CMake configs
├── __legacy__matlab-labchart-signals.zip # Archived per-subject MATLAB LabChart signal parsing scripts
└── __legacy__matlab-meshing-pipeline.zip # Archived original MATLAB Iso2Mesh pipeline
```

---

## Context & Superseding Modules

* **`pcv_lung.py`**: Earlier pressure-controlled ventilation solver. The current simulator focuses on volume-controlled ventilation (`src/core/vcv_lung.py`), but PCV remains a useful reference for future pressure-limited studies.
* **Nelder-Mead Routines (`nelder-mead.py`, `run-calibration.py`)**: Earlier gradient-free simplex optimization approach. Completely superseded by the Bayesian Optimization framework in `src/core/BOCalibration.py`.
* **Archived ZIP Bundles**: Keep repository size small and prevent file clutter while safeguarding historical multi-language workflows (MATLAB, C++/CGAL).
