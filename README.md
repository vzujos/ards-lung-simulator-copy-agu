# ARDS Lung Simulator

_Computational Medicine Laboratory, Pontificia Universidad Católica de Chile ([@comp-medicine-uc](https://github.com/comp-medicine-uc))_

A biophysically detailed, finite element simulation platform for mechanical ventilation in Acute Respiratory Distress Syndrome (ARDS). The simulator couples 3D poroelastic continuum lung parenchymal mechanics with a multi-generation anatomical airway tree to investigate regional aeration, strain distribution, and mechanical ventilation strategies (ARDSNet low-tidal volume vs. APRV).

---

## Key Features

* **Finite-Deformation Poroelasticity**: Implemented in **FEniCS** (`dolfin`), solving coupled non-linear equations for solid tissue displacement $\mathbf{u}$ and pore pressure $p$.
* **Airway Fluid Network**: Object-oriented tree modeling (`AirwayManager`) accounting for anatomical geometry, Reynolds-dependent flow resistances, and terminal flow splitting.
* **Nonlinear Hyperelastic Constitutive Laws**: Compressible porous tissue formulations (Birzle, exponential, neo-Hookean) capturing progressive stiffening and regional compliance.
* **Bayesian Optimization Calibration**: Automated parameter estimation (`BOCalibration`) fitting tissue stiffness and permeability against measured clinical and animal ventilator waveforms.
* **Scientific Manuscript Reproducibility**: Complete suite of figure generation scripts, plotting style libraries, and compiled publication outputs in `manuscript/`.

---

## Repository Structure

```
ards-lung-simulator/
├── src/
│   ├── core/           # Simulation engine (vcv_lung, AirwayManager, modelfunctions, BOCalibration)
│   ├── meshing/        # 3-step meshing pipeline (segmentation -> surface BC tagging -> FEniCS mesh)
│   ├── postprocessing/ # Results extraction (VTU grouping, biomechanical metrics, validation)
│   ├── experiment/     # Staging area for exploratory models and analysis under review
│   ├── legacy/         # Preserved historical solvers (PCV), early calibration, and archives
│   └── trash/          # [Temporary] Staged obsolete files slated for manual deletion
│
├── scripts/            # Command-line entrypoints for simulation, calibration, and sensitivity
├── manuscript/         # Publication figure generators (code/) and high-res vector figures (figures/)
├── raw-data/           # Animal experimental monitoring records and ventilator signals (Signals/)
├── results-data/       # Output directories for simulation VTU checkpoints and metrics
├── testing-data/       # Reference segmentations (NIfTI), benchmark meshes, and stable test cases
└── presentations/      # Conference talks, slide decks, and project presentation assets
```

---

## Getting Started

### 1. Prerequisites & Environment

The simulator is built on Python 3.9+ and FEniCS (DOLFIN 2019.1.0):

* **Finite Element Solver**: [`FEniCS 2019.1.0`](https://fenicsproject.org/)
* **Scientific Computing**: `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-optimize`, `scikit-learn`
* **Mesh & Imaging**: `meshio`, `nibabel`, `pyvista`, `trimesh`, `iso2mesh`

### 2. Running a Forward Simulation

Launch forward volume-controlled simulations using the universal executer:

```bash
# Run simulation for Pig 5 using a medium-coarse tetrahedral mesh
python scripts/2025-12_universal-executer.py -pig_id 5 -mesh_type medium-coarse
```

### 3. Running Parameter Calibration

Calibrate tissue properties against measured airway pressure waveforms using Bayesian Optimization:

```bash
python scripts/calibrate.py
```

### 4. Reproducing Manuscript Figures

To be included.

---

## License & Citation

Developed by the Computational Medicine Laboratory at Pontificia Universidad Católica de Chile. Refer to the project publication for citation guidelines.



## Cambios al repositorio original (Antes de Pull)

Se descartaron las siguientes carpetas y archivos (por recomendación de Agustín):

- `src/legacy/`
- `src/signal-processing/`
- `src/test/`
- `functions2.py`

Se realizaron los siguientes cambios al archivo `src/execution/2025-12_universal-executer.py`:

- Se cambiaron unas ruas absolutas de las geometrías a rutas relativas (lines 8, 404-410, 504)