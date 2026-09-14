# Scientific Manuscript & Reproducibility Suite (`manuscript/`)

This directory contains the complete reproduction package for the scientific manuscript associated with the ARDS Lung Simulator.

## Directory Structure

* **[`code/`](code/)**: All Python figure generation scripts, LaTeX table formatters, and publication plotting style libraries used to produce the paper's visuals.
* **[`figures/`](figures/)**: High-resolution vector (`.pdf`) and raster (`.tiff`) publication figures referenced in the manuscript.

---

## Reproducing Manuscript Figures & Tables

All figures presented in the paper can be reproduced from simulation results using the scripts in `manuscript/code/`:

| Manuscript Figure | Generator Script | Primary Output |
| :--- | :--- | :--- |
| Global Respiratory Mechanics Panel | `code/figure_glob-resp-mech-panel.py` | `figures/grm-panel.pdf` |
| Time-Resolved Flow & Pressure Waves | `code/figure_glob-resp-mech-signals.py` | `figures/grm-signals.pdf` |
| Regional Aeration & Ventilation Panels | `code/figure_reg-resp-mech-panels.py` | `figures/reg-aer-comp-panel.pdf`<br>`figures/reg-ventilation-panel.pdf` |
| 2D ROI Aeration & Ventilation Matrices | `code/figure_roi_matrices.py` | `figures/reg2D-aercomp-panel.pdf`<br>`figures/representative-subject2.pdf` |
| Parameter Sensitivity Analysis | `code/figure_sensibility-2.py` | `figures/sensibility-2.pdf` |
| Clinical Sensitivity Metrics | `code/figure_sensitivity-clinical.py` | `figures/sensibility.pdf` |
| Summary Respiratory Mechanics Table | `code/table_generator.py` | LaTeX table snippet |

Refer to [`manuscript/code/README.md`](code/README.md) and [`manuscript/figures/README.md`](figures/README.md) for detailed descriptions of each figure and associated datasets.
