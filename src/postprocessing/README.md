# Postprocessing & Validation Pipeline (`src/postprocessing/`)

This directory contains tools for processing raw finite element simulation outputs (VTU/PVD/XDMF files), mapping physical fields across anatomical regions, and calculating validation metrics against experimental measurements.

## Module Index

```
src/postprocessing/
├── simulation-vtu-grouper.py     # VTU output aggregator & reference geometry projection
├── biomechanical-analysis.py     # Regional time series extraction & biomechanical metric assembly
├── determine-means.py            # Cohort-wide statistical aggregation across conditions
└── evaluate_simulations.py       # Global & regional quantitative error evaluation
```

---

## Script Descriptions

### 1. `simulation-vtu-grouper.py` (formerly `advanced_vtu_grouper.py`)
* Processes temporal sequences of unstructured VTU output files generated during `vcv_lung.py` runs.
* Retrieves the unloaded reference geometry and projects deformation gradients ($\mathbf{F}$), local Jacobian determinants ($J$), delta gas fractions ($\Delta \text{GF}$), and pore pressures ($p$) back onto consistent anatomical meshes.
* Generates unified VTU datasets for 3D ParaView visualization and regional slicing.

### 2. `biomechanical-analysis.py` (formerly `2025-05-14_BiomechAnalysis_Management.py`)
* Aggregates simulation outputs across distinct anatomical compartments (dependent vs. non-dependent zones, ventral vs. dorsal, cranial vs. caudal).
* Extracts time-resolved pressure-volume curves, tissue stresses, and tidal volume distribution.
* Prepares tabular datasets used by downstream publication figure scripts and statistical models.

### 3. `determine-means.py`
* Calculates statistical metrics (means, standard deviations, confidence intervals) across animal subjects and ventilation regimes (e.g., ARDSnet vs. APRV).
* Computes aggregate aeration distribution percentages across the four standard aeration states (hyperaerated, normoaerated, poorly aerated, non-aerated).

### 4. `evaluate_simulations.py`
* Quantitative validation pipeline assessing simulation accuracy:
  * Computes errors between simulated airway pressures and measured experimental waveforms (peak inspiratory pressure, plateau pressure, PEEP).
  * Evaluates global elastance ($E_{rs}$) and airway resistance ($R_{aw}$).
  * Computes regional volumetric error metrics against end-inspiratory and end-expiratory CT scans.
