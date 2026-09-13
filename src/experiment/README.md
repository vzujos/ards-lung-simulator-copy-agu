# Experimental & Staging Sandbox (`src/experiment/`)

This directory serves as an exploratory staging area for candidate numerical models, image analysis tools, and algorithmic prototypes that are currently retained for evaluation and future development.

## Directory Contents

```
src/experiment/
├── 2026-03_compare_gvs_from_meshes_and_images.py # Gas volume distribution comparison (FE mesh vs. CT)
├── 2026-01_groupwise-image-intensity-analysis.py # Cohort-wide CT Hounsfield Unit attenuation analyzer
├── 2026-01_jacobian-and-dgf.py                  # Correlates FE Jacobian determinant with delta gas fraction
├── 2025-10_constitutive-model.py                 # Formulation and stress-strain testing for porous laws
├── 2024-12-13_mamodel_test_modified.py          # Modified MA constitutive model exploration
├── 2025-05-12_IoU_Study.py                       # 3D Intersection over Union geometric registration study
├── visualize-ROI-partitions.py                   # PyVista 3D interactive viewer for regional partition bounds
├── AwMngr_workbench.py                           # Interactive workbench for airway graph manipulations
├── AirwayHelper.py                               # Auxiliary centerline and terminal point extraction routines
├── test_sigmoid_function_for_stiffness.py        # Mathematical validation of sigmoid variable stiffness
├── standalone_signal_visualizer.py               # Standalone Matplotlib GUI viewer for ventilation waveforms
├── evaluate_simulations_.py                      # Staged evaluation variant under review
├── evaluate_simulations_2.py                     # Second staged evaluation variant under review
└── __legacy__inverse-problem-testlab.zip         # Archived laboratory for inverse poroelastic test cases
```

---

## Highlights of Retained Scripts

* **Mesh-to-Image CT Comparison (`2026-03_compare_gvs_...py`)**: Validates that local volume changes predicted by finite element elements correspond to quantitative aeration shifts observed in matched inspiratory-expiratory CT scans.
* **CT Attenuation Analysis (`2026-01_groupwise-...py`)**: Analyzes tissue density histograms across subjects in Hounsfield units (HU) to benchmark lung recruitability and non-aerated tissue fractions.
* **Kinematics & Gas Fraction Correlation (`2026-01_jacobian-...py`)**: Evaluates the theoretical relationship between the deformation gradient determinant $J = \det(\mathbf{F})$ and changes in regional gas volume fraction.
* **Archived Inverse Problem Lab (`__legacy__inverse-problem-testlab.zip`)**: Contains the series of numerical benchmark problems (1 to 17) used to prototype forward and inverse poroelastic formulations on idealized cubes and lung geometries.
