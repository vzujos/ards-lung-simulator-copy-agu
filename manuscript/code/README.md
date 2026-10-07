# Manuscript Figure Generators & Plotting Libraries (`manuscript/code/`)

This directory contains the complete suite of Python scripts and visualization modules used to postprocess simulation outputs, assemble multi-panel comparative figures, and generate LaTeX tables for the scientific manuscript.

## Scripts & Visualization Modules

```
manuscript/code/
├── figure_glob-resp-mech-panel.py   # Multi-panel global mechanics comparison (simulated vs measured)
├── figure_glob-resp-mech-signals.py # Waveform traces of airway pressure and flow over time
├── figure_reg-resp-mech-panels.py   # Regional aeration compartment volume fractions & ventilation
├── figure_roi_matrices.py           # 2D coronal/sagittal ROI aeration and ventilation heatmaps
├── figure_sensibility-2.py          # Multi-parameter sensitivity response curves
├── figure_sensibility.py            # Earlier baseline sensitivity figure generator
├── figure_sensitivity-clinical.py   # Sensitivity of clinical indices (elastance, driving pressure)
├── table_generator.py               # Formats global mechanics metrics into publication LaTeX tables
├── snapshots-processor.py           # Batch cropping tool for 3D ParaView renderings
├── new_plotter.py                   # Plotting styling, color palettes, and multi-panel layout engine
├── ROIAnalysis.py                   # Slicing and spatial region-of-interest partitioning logic
├── prepare_bma.py                   # Assembles raw simulation outputs into biomechanical arrays
└── ParaviewProgrammableSource_VisualizeAirwayTree.txt # ParaView source script for 3D airway visualization
```

---

## Script Descriptions & Outputs

### 1. `figure_glob-resp-mech-panel.py`
* Assembles multi-panel boxplots and scatter plots comparing global respiratory mechanics:
  * Static elastance ($E_{stat}$) and dynamic elastance ($E_{dyn}$)
  * Airway resistance ($R_{aw}$) and peak airway pressure ($P_{peak}$)
  * Driving pressure ($\Delta P$) and end-inspiratory plateau pressure ($P_{plat}$)
* **Outputs**: `figures/grm-panel.pdf`, `figures/grm-panel.tiff`, `figures/grm-panel-reduced.pdf`.

### 2. `figure_glob-resp-mech-signals.py`
* Plots continuous time-resolved waveforms comparing measured ventilator curves against forward finite element predictions across inspiration, pause, and expiration.
* **Output**: `figures/grm-signals.pdf`.

### 3. `figure_reg-resp-mech-panels.py`
* Computes regional aeration compartment fractions across lung gravitational zones (ventral, middle, dorsal).
* Shows shifts between non-aerated (atelectatic), poorly aerated, normally aerated, and hyperaerated tissue.
* **Outputs**: `figures/reg-aer-comp-panel.pdf`, `figures/reg-ventilation-panel.pdf`.

### 4. `figure_roi_matrices.py`
* Generates 2D regional grid heatmaps representing spatial ventilation distribution and aeration shifts across coronal and sagittal slices for representative animals.
* Requires `ROIAnalysis.py` and `BidimLib.py`.
* **Outputs**: `figures/reg2D-aercomp-panel.pdf`, `figures/reg2D-ventilation-panel.pdf`, `figures/representative-subject*.pdf`.

### 5. `figure_sensibility-2.py` & `figure_sensitivity-clinical.py`
* Visualizes model sensitivity to changes in solid matrix stiffness ($c_1$), fiber stiffness ($c_2$), baseline permeability ($k_0$), and peripheral airway resistance scaling.
* **Outputs**: `figures/sensibility-2.pdf`, `figures/sensibility.pdf`.

### 6. `table_generator.py`
* Reads cohort results data and automatically outputs formatted LaTeX table rows summarizing mean and standard deviation values for physiological and mechanical parameters.
