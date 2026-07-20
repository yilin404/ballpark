"""Deterministic per-link quality metrics for collision-sphere fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from scipy.stats import qmc

from ._spherize import Sphere
from ._warp_mesh_query import WarpMeshQuery


_METRIC_COVERAGE_SAMPLE_SEED = 44


@dataclass(frozen=True)
class SphereFitMetrics:
    """Quality metrics for spheres fitted to one collision mesh.

    ``coverage`` and ``protrusion`` use the closed interval ``[0, 1]``.
    ``volume_ratio`` is non-negative and may exceed ``1`` when the total sphere
    volume exceeds the mesh volume. All distance fields use meters.
    """

    num_spheres: int = 0
    coverage: float = 0.0
    protrusion: float = 0.0
    protrusion_dist_mean: float = 0.0
    protrusion_dist_p95: float = 0.0
    surface_gap_mean: float = 0.0
    surface_gap_p95: float = 0.0
    max_uncovered_gap: float = 0.0
    volume_ratio: float = 0.0
    min_radius_ratio: float = 0.0
    radius_ratio_p05: float = 0.0
    small_sphere_fraction: float = 0.0
    unique_coverage_contribution: float = 0.0
    redundant_sphere_fraction: float = 0.0
    effective_sphere_count: int = 0
    overlap_pair_fraction: float = 0.0
    overlap_penetration_p95_ratio: float = 0.0


def sample_mesh_surface(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    """Generate deterministic area-weighted samples on a triangle mesh.

    Args:
        mesh: Source triangle mesh in link-local coordinates.
        count: Positive number of requested surface samples.

    Returns:
        Array of shape ``(count, 3)`` in meters.

    Raises:
        ValueError: If ``count`` is non-positive or the mesh has no area.
    """
    if count <= 0:
        raise ValueError("surface sample count must be positive")
    surface_points, _ = _sample_mesh_surface_with_faces(mesh, count)
    return surface_points


def _sample_mesh_surface_with_faces(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic area-weighted points and source face indices.

    Args:
        mesh: Source triangle mesh in link-local coordinates.
        count: Positive number of requested surface samples.
        seed: Scrambling seed for deterministic Halton sampling.

    Returns:
        Surface points with shape ``(count, 3)`` and face indices with shape
        ``(count,)``.

    Raises:
        ValueError: If ``count`` is non-positive or the mesh has no area.
    """
    if count <= 0:
        raise ValueError("surface sample count must be positive")
    triangle_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total_area = float(triangle_areas.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("mesh must contain triangles with positive area")

    samples = qmc.Halton(d=3, scramble=True, seed=seed).random(n=count)
    face_positions = samples[:, 0] * total_area
    face_indices = np.searchsorted(np.cumsum(triangle_areas), face_positions, side="right")
    selected_triangles = np.asarray(mesh.triangles)[face_indices]
    sqrt_u = np.sqrt(samples[:, 1])
    barycentric = np.stack(
        (
            1.0 - sqrt_u,
            sqrt_u * (1.0 - samples[:, 2]),
            sqrt_u * samples[:, 2],
        ),
        axis=1,
    )
    surface_points = np.einsum("nij,ni->nj", selected_triangles, barycentric)
    return surface_points, face_indices


def sample_sphere_directions(count: int) -> np.ndarray:
    """Generate deterministic approximately uniform unit-sphere directions.

    Args:
        count: Positive number of requested directions.

    Returns:
        Array of shape ``(count, 3)`` containing unit vectors.

    Raises:
        ValueError: If ``count`` is non-positive.
    """
    if count <= 0:
        raise ValueError("sphere direction count must be positive")
    sample_indices = np.arange(count, dtype=np.float64) + 0.5
    z_coordinates = 1.0 - 2.0 * sample_indices / count
    radial = np.sqrt(np.maximum(0.0, 1.0 - z_coordinates**2))
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * sample_indices
    return np.stack(
        (radial * np.cos(azimuth), radial * np.sin(azimuth), z_coordinates),
        axis=1,
    )


def sample_mesh_coverage_points(
    mesh: trimesh.Trimesh,
    mesh_query: WarpMeshQuery,
    count: int,
    device: torch.device,
    seed: int = 43,
) -> torch.Tensor:
    """Sample deterministic volume and inward-offset surface points.

    Args:
        mesh: Original collision mesh.
        mesh_query: Warp SDF query for the same mesh.
        count: Target number of coverage samples.
        device: Torch device used for SDF filtering.
        seed: Scrambling seed for deterministic Halton sampling.

    Returns:
        Coverage sample tensor of shape ``(count, 3)``.
        Watertight meshes use both volume and inset-surface points. Other
        meshes use SDF-validated inset points. Rejected interior or inset
        candidates are replaced by deterministic surface points.

    Raises:
        ValueError: If ``count`` is non-positive.
    """
    if count <= 0:
        raise ValueError("coverage sample count must be positive")

    sample_parts: list[torch.Tensor] = []
    volume_count = count // 2

    # Sample bounding-box candidates, then retain only mesh-interior volume points.
    if mesh.is_watertight:
        unit_samples = qmc.Halton(d=3, scramble=True, seed=seed).random(
            n=max(volume_count * 8, 1)
        )
        box_samples = qmc.scale(unit_samples, mesh.bounds[0], mesh.bounds[1])
        box_tensor = torch.as_tensor(box_samples, dtype=torch.float32, device=device)
        signed_distance, _ = mesh_query.query_sdf(box_tensor)
        interior = box_tensor[signed_distance <= 0.0]
        if interior.shape[0] > 0:
            sample_parts.append(interior[:volume_count])

    # Offset deterministic surface samples inward and validate them against the SDF.
    inset_target = count - sum(part.shape[0] for part in sample_parts)
    surface_points, face_indices = _sample_mesh_surface_with_faces(
        mesh,
        max(inset_target, 1),
        seed=seed,
    )
    mesh_extent = float(np.linalg.norm(mesh.extents))
    inset_points = surface_points - mesh.face_normals[face_indices] * (
        mesh_extent * 0.005
    )
    inset_tensor = torch.as_tensor(
        inset_points,
        dtype=torch.float32,
        device=device,
    )
    inset_signed_distance, _ = mesh_query.query_sdf(inset_tensor)
    validated_inset = inset_tensor[inset_signed_distance <= 0.0]
    if validated_inset.shape[0] > 0:
        sample_parts.append(validated_inset[:inset_target])

    # Fill every rejected SDF candidate from the same deterministic surface set.
    sampled_count = sum(part.shape[0] for part in sample_parts)
    remaining_count = count - sampled_count
    if remaining_count > 0:
        sample_parts.append(
            torch.as_tensor(
                surface_points[:remaining_count],
                dtype=torch.float32,
                device=device,
            )
        )
    return torch.cat(sample_parts, dim=0)


def compute_sphere_fit_metrics(
    mesh: trimesh.Trimesh,
    spheres: list[Sphere],
    device: torch.device | None = None,
    n_coverage: int = 2048,
    n_surface: int = 2048,
    n_sphere_surface: int = 128,
) -> SphereFitMetrics:
    """Compute deterministic metrics against the original collision mesh.

    Args:
        mesh: Original link-local collision mesh in meters.
        spheres: Link-local fitted spheres.
        device: Optional Torch device; defaults to CUDA when available and CPU
            otherwise while retaining the same Warp kernels.
        n_coverage: Positive number of interior or fallback surface coverage
            samples.
        n_surface: Positive number of mesh-surface gap samples.
        n_sphere_surface: Positive number of surface directions per sphere.

    Returns:
        Complete finite quality metrics for a non-empty sphere list.

    Raises:
        ValueError: If no spheres are supplied, sphere values are invalid, or
            any sampling count is non-positive.

    Note:
        This function initializes and uses the Torch and Warp runtimes on the
        selected compute device.
    """
    # Validate and pack the fitted sphere geometry on the selected compute device.
    if not spheres:
        raise ValueError("sphere metrics require at least one sphere")
    torch_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centers = torch.as_tensor(
        np.stack([np.asarray(sphere.center, dtype=np.float32) for sphere in spheres]),
        dtype=torch.float32,
        device=torch_device,
    )
    radii = torch.as_tensor(
        [float(sphere.radius) for sphere in spheres],
        dtype=torch.float32,
        device=torch_device,
    )
    if not torch.isfinite(centers).all() or not torch.isfinite(radii).all() or torch.any(radii <= 0.0):
        raise ValueError("sphere centers and positive radii must be finite")

    # Measure how much of the deterministic mesh coverage set lies inside a sphere.
    mesh_query = WarpMeshQuery(mesh, torch_device)
    coverage_points = sample_mesh_coverage_points(
        mesh,
        mesh_query,
        n_coverage,
        torch_device,
        seed=_METRIC_COVERAGE_SAMPLE_SEED,
    )
    coverage_distances = torch.cdist(coverage_points, centers) - radii.unsqueeze(0)
    sample_sphere_membership = coverage_distances <= 0.0
    coverage_count_per_sample = sample_sphere_membership.sum(dim=1)
    coverage = float((coverage_count_per_sample > 0).float().mean())

    # Measure relative sphere size and exclusive contribution to the fixed
    # coverage set. A sphere is effective only when it exclusively covers at
    # least one sampled point; this definition intentionally avoids inferring
    # continuous-geometry necessity from a finite sample set.
    mesh_extent = float(np.linalg.norm(mesh.extents))
    if not np.isfinite(mesh_extent) or mesh_extent <= 0.0:
        raise ValueError("sphere metric mesh must have positive finite extent")
    radius_ratio = radii / mesh_extent
    unique_sample_mask = coverage_count_per_sample == 1
    unique_coverage_contribution = float(unique_sample_mask.float().mean())
    unique_count_per_sphere = (
        sample_sphere_membership & unique_sample_mask.unsqueeze(1)
    ).sum(dim=0)
    effective_sphere_mask = unique_count_per_sphere > 0
    effective_sphere_count = int(effective_sphere_mask.sum().item())
    redundant_sphere_fraction = float((~effective_sphere_mask).float().mean())

    # Quantify pairwise overlap using each unordered sphere pair exactly once.
    # The single-sphere and no-overlap cases deliberately return finite zeros.
    sphere_count = centers.shape[0]
    overlap_pair_fraction = 0.0
    overlap_penetration_p95_ratio = 0.0
    if sphere_count >= 2:
        pair_indices = torch.triu_indices(
            sphere_count,
            sphere_count,
            offset=1,
            device=torch_device,
        )
        pair_distance = torch.linalg.vector_norm(
            centers[pair_indices[0]] - centers[pair_indices[1]],
            dim=1,
        )
        pair_penetration_ratio = torch.relu(
            radii[pair_indices[0]] + radii[pair_indices[1]] - pair_distance
        ) / mesh_extent
        positive_overlap_mask = pair_penetration_ratio > 0.0
        overlap_pair_fraction = float(positive_overlap_mask.float().mean())
        if torch.any(positive_overlap_mask):
            overlap_penetration_p95_ratio = float(
                torch.quantile(pair_penetration_ratio[positive_overlap_mask], 0.95)
            )

    # Query sampled sphere boundaries to quantify protrusion outside the source mesh.
    directions = torch.as_tensor(
        sample_sphere_directions(n_sphere_surface),
        dtype=torch.float32,
        device=torch_device,
    )
    sphere_surface = centers[:, None, :] + radii[:, None, None] * directions[None, :, :]
    sphere_sdf, _ = mesh_query.query_sdf(sphere_surface.reshape(-1, 3))
    outside_distance = torch.relu(sphere_sdf)
    outside_mask = outside_distance > 0.0
    protrusion = float(outside_mask.float().mean())
    if torch.any(outside_mask):
        positive_distance = outside_distance[outside_mask]
        protrusion_mean = float(positive_distance.mean())
        protrusion_p95 = float(torch.quantile(positive_distance, 0.95))
    else:
        protrusion_mean = 0.0
        protrusion_p95 = 0.0

    # Measure uncovered mesh-surface gaps and aggregate scale-aware volume metrics.
    surface_points = torch.as_tensor(
        sample_mesh_surface(mesh, n_surface),
        dtype=torch.float32,
        device=torch_device,
    )
    surface_gaps = torch.relu(
        (torch.cdist(surface_points, centers) - radii.unsqueeze(0)).min(dim=1).values
    )
    sphere_volume = float((4.0 / 3.0 * np.pi * (radii**3).sum()).item())
    mesh_volume = (
        float(abs(mesh.volume))
        if mesh.is_watertight and abs(float(mesh.volume)) > 1e-12
        else float(np.prod(mesh.extents))
    )
    return SphereFitMetrics(
        num_spheres=len(spheres),
        coverage=coverage,
        protrusion=protrusion,
        protrusion_dist_mean=protrusion_mean,
        protrusion_dist_p95=protrusion_p95,
        surface_gap_mean=float(surface_gaps.mean()),
        surface_gap_p95=float(torch.quantile(surface_gaps, 0.95)),
        max_uncovered_gap=float(surface_gaps.max()),
        volume_ratio=sphere_volume / max(mesh_volume, 1e-12),
        min_radius_ratio=float(radius_ratio.min()),
        radius_ratio_p05=float(torch.quantile(radius_ratio, 0.05)),
        small_sphere_fraction=float((radius_ratio < 0.01).float().mean()),
        unique_coverage_contribution=unique_coverage_contribution,
        redundant_sphere_fraction=redundant_sphere_fraction,
        effective_sphere_count=effective_sphere_count,
        overlap_pair_fraction=overlap_pair_fraction,
        overlap_penetration_p95_ratio=overlap_penetration_p95_ratio,
    )
