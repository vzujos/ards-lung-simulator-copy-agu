# Results & Simulation Data (`results-data/`)

This directory is the designated output destination for forward finite element simulation results, Bayesian optimization parameter checkpoints, and processed regional mechanics data.

## Expected Subdirectory Structure

When running simulation runs and calibrations via `scripts/`, output files are saved here following standard directory conventions:

```
results-data/
├── VTK/                     # Time-series spatial field outputs (.pvd, .vtu) for 3D visualization
│   ├── Displacement/        # Solid displacement vector field u
│   ├── Pressure/            # Pore pressure scalar field p
│   ├── DeltaGasFraction/    # Change in local gas volume fraction
│   └── Stress/              # Cauchy and 1st/2nd Piola-Kirchhoff stress fields
│
├── post/                    # Grouped and projected VTU files from simulation-vtu-grouper.py
├── checkpoints/             # Optimization surrogate model states and parameter logs (.csv, .pkl)
└── tables/                  # Summary metrics, elastance, resistance, and volume statistics
```

> [!NOTE]
> Large binary simulation outputs (e.g. detailed 3D VTU files spanning multiple respiratory cycles) are typically excluded from Git tracking via `.gitignore` to maintain repository performance.
