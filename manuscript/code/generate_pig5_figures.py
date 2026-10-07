"""Generate the PIG5 figures supported by the data in this repository.

The outputs are intentionally simulation/local-data figures. Experimental
signals, cohort comparisons, and sensitivity sweeps are not available here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyvista as pv
from matplotlib.colors import BoundaryNorm
from matplotlib.axes import Axes


PIG5_ROOT = Path(__file__).resolve().parents[2]


def triangulate_projected_mesh(points: np.ndarray) -> mtri.Triangulation:
    """Create a projected triangulation without filling large geometric gaps."""
    triangulation = mtri.Triangulation(points[:, 0], points[:, 2])
    projected_points = points[:, [0, 2]]
    triangles = projected_points[triangulation.triangles]
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1),
        ),
        axis=1,
    )
    max_edge_length = np.quantile(edge_lengths.max(axis=1), 0.98)
    triangulation.set_mask(edge_lengths.max(axis=1) > max_edge_length)
    return triangulation


def load_signals(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signals = root / "outputs" / "PIG5-mc-per" / "Signals"
    times = np.load(signals / "effectivetimes.npy").reshape(-1)
    flow = np.load(signals / "fluxes.npy").reshape(-1)
    volume = np.load(signals / "volumenes.npy").reshape(-1)
    pressure = np.load(signals / "presionestodas.npy").reshape(-1) * 10.1972

    size = min(map(len, (times, flow, volume, pressure)))
    return times[:size], flow[:size], volume[:size] - volume[0], pressure[:size]


def generate_signal_figure(root: Path, output_dir: Path) -> Path:
    times, flow, volume, pressure = load_signals(root)
    output = output_dir / "pig5-simulation-signals.pdf"

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.4), sharex=True)
    axes[0].plot(times, flow, color="#176b87", linewidth=1.2)
    axes[0].set_ylabel("Air flow\n(L/s)")
    axes[1].plot(times, volume, color="#d16b35", linewidth=1.2)
    axes[1].set_ylabel("Lung volume\n(relative)")
    axes[2].plot(times, pressure, color="#3f6b45", linewidth=1.2)
    axes[2].set_ylabel("Airway pressure\n(cmH$_2$O)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(0, pressure.max() * 1.1)

    for axis in axes:
        axis.grid(axis="y", color="#d9dedb", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        axis.set_xlim(0, times.max())
        axis.axvline(0.75, color="#797979", linewidth=0.6)
        axis.axvline(0.375, color="#797979", linewidth=0.6)


    fig.suptitle("PIG5 respiratory mechanics: simulation", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_field(axis: Axes, mesh: pv.DataSet, field: str, title: str) -> None:
    values = np.asarray(mesh.point_data[field]).reshape(-1)
    points = np.asarray(mesh.points)
    triangulation = triangulate_projected_mesh(points)
    field_plot = axis.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap="Blues",
        linewidth=0,
    )
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("z")
    axis.tick_params(labelsize=7)
    plt.colorbar(field_plot, ax=axis, fraction=0.046, pad=0.04)


def plot_field_scatter(
    axis: Axes, mesh: pv.DataSet, field: str, title: str
) -> None:
    values = np.asarray(mesh.point_data[field]).reshape(-1)
    points = np.asarray(mesh.points)
    field_plot = axis.scatter(
        points[:, 0],
        points[:, 2],
        c=values,
        s=10,
        cmap="Blues",
        alpha=0.7,
        linewidths=0,
    )
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("z")
    axis.tick_params(labelsize=7)
    plt.colorbar(field_plot, ax=axis, fraction=0.046, pad=0.04)


def plot_aeration_categories(
    axis: Axes,
    mesh: pv.DataSet,
    field: str,
    title: str,
    colorbar_axis: Axes,
) -> None:
    values = np.asarray(mesh.point_data[field]).reshape(-1)
    points = np.asarray(mesh.points)
    bounds = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    category_names = ["NAT", "PAT", "AT", "HIT"]
    cmap = plt.get_cmap("Blues", len(category_names))
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    triangulation = triangulate_projected_mesh(points)
    categories = np.digitize(
        np.clip(values, bounds[0], bounds[-1]), bounds[1:-1]
    )
    triangle_categories = np.array(
        [
            np.bincount(categories[triangle], minlength=len(category_names)).argmax()
            for triangle in triangulation.triangles
        ]
    )

    category_plot = axis.tripcolor(
        triangulation,
        facecolors=triangle_categories,
        shading="flat",
        cmap=cmap,
        norm=norm,
    )
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("z")
    axis.tick_params(labelsize=7)

    colorbar = plt.colorbar(
        category_plot,
        cax=colorbar_axis,
        boundaries=bounds,
        ticks=(bounds[:-1] + bounds[1:]) / 2,
    )
    colorbar.ax.set_yticklabels(category_names)
    colorbar.set_label("Categoría HU")


def plot_aeration_categories_scatter(
    axis: Axes,
    mesh: pv.DataSet,
    field: str,
    title: str,
    colorbar_axis: Axes,
) -> None:
    values = np.asarray(mesh.point_data[field]).reshape(-1)
    points = np.asarray(mesh.points)
    bounds = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    category_names = ["NAT", "PAT", "AT", "HIT"]
    cmap = plt.get_cmap("Blues", len(category_names))
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    category_plot = axis.scatter(
        points[:, 0],
        points[:, 2],
        c=np.clip(values, bounds[0], bounds[-1]),
        s=10,
        alpha=0.7,
        cmap=cmap,
        norm=norm,
        linewidths=0,
    )
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("z")
    axis.tick_params(labelsize=7)
    colorbar = plt.colorbar(
        category_plot,
        cax=colorbar_axis,
        boundaries=bounds,
        ticks=(bounds[:-1] + bounds[1:]) / 2,
    )
    colorbar.ax.set_yticklabels(category_names)
    colorbar.set_label("Categoría HU")


def plot_transition_categories(
    axis: Axes,
    mesh: pv.DataSet,
    title: str,
    colorbar_axis: Axes,
) -> None:
    end_expiratory = np.asarray(
        mesh.point_data["End-Expiratory Porosity"]
    ).reshape(-1)
    end_inspiratory = np.asarray(
        mesh.point_data["End-Inspiratory Porosity"]
    ).reshape(-1)
    points = np.asarray(mesh.points)

    category_bounds = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    categories = np.digitize(
        np.clip(np.stack((end_expiratory, end_inspiratory)), 0.0, 1.0),
        category_bounds[1:-1],
    )
    start, end = categories
    transition_values = np.array(
        [
            [0, 1, 2, 3],
            [-1, 0, 1, 2],
            [-2, -1, 0, 1],
            [-3, -2, -1, 0],
        ]
    )
    values = transition_values[start, end]
    transition_bounds = np.arange(-3.5, 4.5, 1.0)
    cmap = plt.get_cmap("RdBu", 7)
    norm = BoundaryNorm(transition_bounds, cmap.N)
    triangulation = triangulate_projected_mesh(points)
    triangle_values = np.array(
        [
            np.median(values[triangle])
            for triangle in triangulation.triangles
        ]
    )

    transition_plot = axis.tripcolor(
        triangulation,
        facecolors=triangle_values,
        shading="flat",
        cmap=cmap,
        norm=norm,
    )
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("z")
    axis.tick_params(labelsize=7)

    colorbar = plt.colorbar(
        transition_plot,
        cax=colorbar_axis,
        boundaries=transition_bounds,
        ticks=np.arange(-3, 4),
    )
    colorbar.set_label("Cambio de categoría")


def plot_transition_categories_scatter(
    axis: Axes,
    mesh: pv.DataSet,
    title: str,
    colorbar_axis: Axes,
) -> None:
    end_expiratory = np.asarray(
        mesh.point_data["End-Expiratory Porosity"]
    ).reshape(-1)
    end_inspiratory = np.asarray(
        mesh.point_data["End-Inspiratory Porosity"]
    ).reshape(-1)
    points = np.asarray(mesh.points)
    category_bounds = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    categories = np.digitize(
        np.clip(np.stack((end_expiratory, end_inspiratory)), 0.0, 1.0),
        category_bounds[1:-1],
    )
    start, end = categories
    transition_values = np.array(
        [[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1], [-3, -2, -1, 0]]
    )
    values = transition_values[start, end]
    transition_bounds = np.arange(-3.5, 4.5, 1.0)
    transition_plot = axis.scatter(
        points[:, 0],
        points[:, 2],
        c=values,
        s=10,
        alpha=0.7,
        cmap=plt.get_cmap("RdBu", 7),
        norm=BoundaryNorm(transition_bounds, 7),
        linewidths=0,
    )
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("z")
    axis.tick_params(labelsize=7)
    colorbar = plt.colorbar(
        transition_plot,
        cax=colorbar_axis,
        boundaries=transition_bounds,
        ticks=np.arange(-3, 4),
    )
    colorbar.set_label("Cambio de categoría")


def generate_regional_figure(root: Path, output_dir: Path) -> Path:
    mesh_dir = root / "testing-data" / "PIG5" / "ARDSnet" / "medium"
    experiment = pv.read(mesh_dir / "reg_anim_ready.vtu")
    simulation = pv.read(mesh_dir / "sim_anim_ready.vtu")
    fields = (
        "End-Expiratory Porosity",
        "End-Inspiratory Porosity",
        "Delta Porosity",
    )
    output = output_dir / "pig5-regional-fields.pdf"

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), squeeze=False)
    for column, field in enumerate(fields):
        plot_field_scatter(axes[0, column], experiment, field, f"Experiment: {field}")
        plot_field_scatter(axes[1, column], simulation, field, f"Simulation: {field}")
    fig.suptitle("PIG5 regional porosity fields", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def generate_regional_filled_figure(root: Path, output_dir: Path) -> Path:
    mesh_dir = root / "testing-data" / "PIG5" / "ARDSnet" / "medium"
    experiment = pv.read(mesh_dir / "reg_anim_ready.vtu")
    simulation = pv.read(mesh_dir / "sim_anim_ready.vtu")
    fields = (
        "End-Expiratory Porosity",
        "End-Inspiratory Porosity",
        "Delta Porosity",
    )
    output = output_dir / "pig5-regional-fields-filled.pdf"

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), squeeze=False)
    for column, field in enumerate(fields):
        plot_field(axes[0, column], experiment, field, f"Experiment: {field}")
        plot_field(axes[1, column], simulation, field, f"Simulation: {field}")
    fig.suptitle("PIG5 regional porosity fields", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def generate_regional_category_figure(root: Path, output_dir: Path) -> Path:
    mesh_dir = root / "testing-data" / "PIG5" / "ARDSnet" / "medium"
    experiment = pv.read(mesh_dir / "reg_anim_ready.vtu")
    simulation = pv.read(mesh_dir / "sim_anim_ready.vtu")
    fields = (
        "End-Expiratory Porosity",
        "End-Inspiratory Porosity",
    )
    state_names = ("End-expiratory", "End-inspiratory")
    output = output_dir / "pig5-regional-aeration-categories.pdf"

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), squeeze=False)
    for row, (kind, mesh) in enumerate(
        (("Experiment", experiment), ("Simulation", simulation))
    ):
        for column, (field, state_name) in enumerate(zip(fields, state_names)):
            plot_aeration_categories_scatter(
                axes[row, column],
                mesh,
                field,
                f"{kind}: {state_name}",
                axes[row, column].inset_axes((1.03, 0.08, 0.08, 0.84)),
            )
        plot_transition_categories_scatter(
            axes[row, 2],
            mesh,
            f"{kind}: transition",
            axes[row, 2].inset_axes((1.03, 0.08, 0.08, 0.84)),
        )
    fig.suptitle("PIG5 regional aeration categories", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def generate_regional_category_filled_figure(
    root: Path, output_dir: Path
) -> Path:
    mesh_dir = root / "testing-data" / "PIG5" / "ARDSnet" / "medium"
    experiment = pv.read(mesh_dir / "reg_anim_ready.vtu")
    simulation = pv.read(mesh_dir / "sim_anim_ready.vtu")
    fields = (
        "End-Expiratory Porosity",
        "End-Inspiratory Porosity",
    )
    state_names = ("End-expiratory", "End-inspiratory")
    output = output_dir / "pig5-regional-aeration-categories-filled.pdf"

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), squeeze=False)
    for row, (kind, mesh) in enumerate(
        (("Experiment", experiment), ("Simulation", simulation))
    ):
        for column, (field, state_name) in enumerate(zip(fields, state_names)):
            plot_aeration_categories(
                axes[row, column],
                mesh,
                field,
                f"{kind}: {state_name}",
                axes[row, column].inset_axes((1.03, 0.08, 0.08, 0.84)),
            )
        plot_transition_categories(
            axes[row, 2],
            mesh,
            f"{kind}: transition",
            axes[row, 2].inset_axes((1.03, 0.08, 0.08, 0.84)),
        )
    fig.suptitle("PIG5 regional aeration categories", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PIG5_ROOT,
        help="Repository root (default: inferred from this file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/manuscript/figures).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "manuscript" / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        generate_signal_figure(root, output_dir),
        generate_regional_figure(root, output_dir),
        generate_regional_filled_figure(root, output_dir),
        generate_regional_category_figure(root, output_dir),
        generate_regional_category_filled_figure(root, output_dir),
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()