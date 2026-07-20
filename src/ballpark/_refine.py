"""Independent per-link Warp/SDF collision-sphere refinement."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as torch_functional
import trimesh
from loguru import logger

from ._config import RefineParams
from ._metrics import sample_mesh_coverage_points, sample_sphere_directions
from ._spherize import Sphere
from ._warp_mesh_query import WarpMeshQuery, WarpSignedDistance


ClipPlane = tuple[str, float]
_CLIP_AXIS_NORMALS: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}
_REFINE_COVERAGE_SAMPLE_SEED = 43


def _resolve_clip_axis(axis: str) -> np.ndarray:
    """Convert a cuRobo-style clip axis into a unit normal.

    Args:
        axis: Axis string ``x``, ``y``, ``z`` or its negated form.

    Returns:
        Link-local unit normal with shape ``(3,)``.

    Raises:
        ValueError: If the axis is unsupported.
    """
    normalized_axis = axis.lower()
    positive_axis = normalized_axis.removeprefix("-")
    if positive_axis not in _CLIP_AXIS_NORMALS or normalized_axis.count("-") > 1:
        raise ValueError(f"Invalid clip axis '{axis}', expected x, y, z, -x, -y, or -z")
    normal = np.asarray(_CLIP_AXIS_NORMALS[positive_axis], dtype=np.float32)
    return -normal if normalized_axis.startswith("-") else normal


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    """Map positive radii to unconstrained softplus parameters.

    Args:
        values: Strictly positive radius offsets.

    Returns:
        Unconstrained parameters satisfying ``softplus(result) == values`` up
        to floating-point precision.
    """
    return values + torch.log(-torch.expm1(-values))


def _coverage_loss(
    coverage_points: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    mesh_extent: float,
) -> torch.Tensor:
    """Penalize original mesh volume samples outside every sphere.

    Args:
        coverage_points: Fixed link-local interior samples of shape ``(M, 3)``;
            non-watertight meshes use surface samples as a fallback.
        centers: Optimized sphere centers of shape ``(N, 3)``.
        radii: Positive sphere radii of shape ``(N,)``.
        mesh_extent: Positive link bounding-box diagonal in meters.

    Returns:
        Scale-normalized squared coverage loss.
    """
    signed_gaps = torch.cdist(coverage_points, centers) - radii.unsqueeze(0)
    temperature = mesh_extent * 0.01
    soft_minimum = -temperature * torch.logsumexp(
        -signed_gaps / temperature,
        dim=1,
    )
    return torch.mean(torch.relu(soft_minimum) ** 2) / (mesh_extent**2)


def _protrusion_reduction(
    signed_distance: torch.Tensor,
    mesh_extent: float,
) -> torch.Tensor:
    """Apply cuRobo's per-sphere max and cross-sphere soft-max reduction.

    Args:
        signed_distance: Mesh signed distances for sphere surface samples with
            shape ``(N, K)`` in meters; positive values are outside the mesh.
        mesh_extent: Positive link bounding-box diagonal in meters.

    Returns:
        Scale-normalized MorphIt protrusion loss.
    """
    squared_outside_distance = torch.relu(signed_distance) ** 2
    per_sphere_maximum = squared_outside_distance.max(dim=1).values
    temperature = mesh_extent * 0.01
    soft_maximum = temperature * torch.logsumexp(
        per_sphere_maximum / temperature,
        dim=0,
    )
    return soft_maximum / (mesh_extent**2)


def _protrusion_loss(
    centers: torch.Tensor,
    radii: torch.Tensor,
    directions: torch.Tensor,
    mesh_query: WarpMeshQuery,
    mesh_extent: float,
) -> torch.Tensor:
    """Query sphere boundaries and compute cuRobo v2 protrusion.

    Args:
        centers: Optimized sphere centers of shape ``(N, 3)``.
        radii: Positive sphere radii of shape ``(N,)``.
        directions: Fixed unit directions of shape ``(K, 3)``.
        mesh_query: Differentiable Warp query for the original link mesh.
        mesh_extent: Positive link bounding-box diagonal in meters.

    Returns:
        Scale-normalized MorphIt protrusion loss.
    """
    sphere_surface = centers[:, None, :] + radii[:, None, None] * directions[None, :, :]
    signed_distance = WarpSignedDistance.apply(
        sphere_surface.reshape(-1, 3),
        mesh_query,
    ).reshape(centers.shape[0], -1)
    return _protrusion_reduction(signed_distance, mesh_extent)


def _tangency_reduction(
    center_signed_distance: torch.Tensor,
    radii: torch.Tensor,
    mesh_extent: float,
) -> torch.Tensor:
    """Apply cuRobo's sphere-center tangency reduction.

    Args:
        center_signed_distance: Mesh SDF at sphere centers with shape ``(N,)``.
        radii: Positive sphere radii with shape ``(N,)``.
        mesh_extent: Positive link bounding-box diagonal in meters.

    Returns:
        Scale-normalized mean squared internal boundary gap.
    """
    boundary_gap = torch.abs(center_signed_distance) - radii
    return torch.mean(torch.relu(boundary_gap) ** 2) / (mesh_extent**2)


def _tangency_loss(
    centers: torch.Tensor,
    radii: torch.Tensor,
    mesh_query: WarpMeshQuery,
    mesh_extent: float,
) -> torch.Tensor:
    """Query sphere centers and compute cuRobo v2 tangency.

    Args:
        centers: Optimized sphere centers of shape ``(N, 3)``.
        radii: Positive sphere radii of shape ``(N,)``.
        mesh_query: Differentiable Warp query for the original link mesh.
        mesh_extent: Positive link bounding-box diagonal in meters.

    Returns:
        Scale-normalized MorphIt tangency loss.
    """
    center_signed_distance = WarpSignedDistance.apply(centers, mesh_query)
    return _tangency_reduction(
        center_signed_distance,
        radii,
        mesh_extent,
    )


def _overlap_loss(
    centers: torch.Tensor,
    radii: torch.Tensor,
    mesh_extent: float,
) -> torch.Tensor:
    """Reproduce cuRobo v2's full-matrix overlap reduction.

    Args:
        centers: Optimized sphere centers of shape ``(N, 3)``.
        radii: Positive sphere radii of shape ``(N,)``.
        mesh_extent: Positive link bounding-box diagonal in meters.

    Returns:
        Mean pairwise penetration over the full symmetric matrix, normalized
        by link extent. Diagonal elements are masked as in cuRobo v2.
    """
    sphere_count = centers.shape[0]
    pairwise_distance = torch.cdist(centers, centers)
    pairwise_distance = (
        pairwise_distance
        + torch.eye(
            sphere_count,
            dtype=centers.dtype,
            device=centers.device,
        )
        * 1e6
    )
    radius_sum = radii.unsqueeze(1) + radii.unsqueeze(0)
    return torch.mean(torch.relu(radius_sum - pairwise_distance)) / mesh_extent


def _halfplane_loss(
    centers: torch.Tensor,
    radii: torch.Tensor,
    plane_normal: torch.Tensor,
    plane_offset: float,
    buffer: float = 0.02,
) -> torch.Tensor:
    """Penalize sphere boundaries crossing a link-local half-space plane.

    Args:
        centers: Optimized sphere centers of shape ``(N, 3)``.
        radii: Positive sphere radii of shape ``(N,)``.
        plane_normal: Unit normal pointing into the allowed half-space.
        plane_offset: Plane offset in link-local meters.
        buffer: Required clearance from the sphere boundary in meters.

    Returns:
        Mean absolute-meter squared half-plane penetration loss.
    """
    signed_distance = centers @ plane_normal - plane_offset
    clearance = signed_distance - radii - buffer
    return torch.mean(torch.relu(-clearance) ** 2)


def refine_link_spheres(
    mesh: trimesh.Trimesh,
    initial_spheres: list[Sphere],
    params: RefineParams,
    clip_plane: Optional[ClipPlane] = None,
    device: torch.device | None = None,
) -> list[Sphere]:
    """Refine one link's spheres against its original collision mesh.

    The function uses fixed deterministic mesh/sphere samples, a differentiable
    Warp SDF, and Torch AdamW. Coverage, protrusion, tangency, and overlap are
    normalized by the link bounding-box diagonal for scale-consistent per-link
    optimization. The optional half-plane Cost remains in absolute meters and
    requests ``params.clip_plane_buffer`` clearance, which defaults to 0.02 m.

    Args:
        mesh: Original link-local collision mesh in meters.
        initial_spheres: Initial link-local spheres from ``Robot.spherize``.
        params: Per-link optimization settings and cost weights.
        clip_plane: Optional ``(axis, offset_m)`` allowed half-space in the
            link-local frame. The soft Cost uses the configured absolute-meter
            buffer.
        device: Optional Torch device, defaulting to CUDA when available.

    Returns:
        Refined link-local spheres in their original ordering.

    Raises:
        ValueError: If the mesh, spheres, parameters, or clip plane are invalid.
        RuntimeError: If optimization produces a non-finite loss.

    Note:
        This function initializes and uses the Torch and Warp runtimes on the
        selected device, executes the configured optimization, and emits an
        informational log after optimization. Its final hard projection is
        independent of all Cost weights: it moves each violating sphere until
        the sphere boundary reaches the plane boundary, without preserving the
        soft ``clip_plane_buffer`` clearance.
    """
    if not initial_spheres:
        return []
    if params.n_iters < 0 or params.n_samples <= 0:
        raise ValueError("refinement iteration and sample counts are invalid")
    if (
        params.center_lr < 0.0
        or params.radius_lr < 0.0
        or params.grad_clip_norm < 0.0
    ):
        raise ValueError("learning rates and gradient clip norm must be non-negative")
    mesh_extent = float(np.linalg.norm(mesh.extents))
    if not math.isfinite(mesh_extent) or mesh_extent <= 0.0:
        raise ValueError("refinement mesh must have positive finite extent")

    torch_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    initial_centers = torch.as_tensor(
        np.stack(
            [np.asarray(sphere.center, dtype=np.float32) for sphere in initial_spheres]
        ),
        dtype=torch.float32,
        device=torch_device,
    )
    initial_radii = torch.as_tensor(
        [float(sphere.radius) for sphere in initial_spheres],
        dtype=torch.float32,
        device=torch_device,
    )
    if torch.any(initial_radii <= 0.0) or not torch.isfinite(initial_centers).all():
        raise ValueError(
            "initial spheres must contain finite centers and positive radii"
        )

    # AdamW updates dimensionless link-normalized variables so the configured
    # learning rates have consistent geometric meaning across link scales.
    initial_centers_normalized = initial_centers / mesh_extent
    initial_radii_normalized = initial_radii / mesh_extent
    minimum_radius_normalized = params.min_radius / mesh_extent
    centers_normalized = torch.nn.Parameter(initial_centers_normalized.clone())
    radius_offset = torch.clamp(
        initial_radii_normalized - minimum_radius_normalized,
        min=1e-6,
    )
    raw_radii = torch.nn.Parameter(_inverse_softplus(radius_offset))
    optimizer = torch.optim.AdamW(
        [
            {"params": [centers_normalized], "lr": params.center_lr},
            {"params": [raw_radii], "lr": params.radius_lr},
        ],
        weight_decay=0.01,
    )

    # Precompute fixed mesh coverage and sphere-surface samples for every iteration.
    mesh_query = WarpMeshQuery(mesh, torch_device)
    coverage_points = sample_mesh_coverage_points(
        mesh=mesh,
        mesh_query=mesh_query,
        count=params.n_samples,
        device=torch_device,
        seed=_REFINE_COVERAGE_SAMPLE_SEED,
    )
    directions = torch.as_tensor(
        sample_sphere_directions(params.n_sphere_surface_samples),
        dtype=torch.float32,
        device=torch_device,
    )

    # Resolve the optional link-local clip plane once before optimization.
    plane_normal: torch.Tensor | None = None
    plane_offset = 0.0
    if clip_plane is not None:
        clip_axis, raw_plane_offset = clip_plane
        plane_normal = torch.as_tensor(
            _resolve_clip_axis(clip_axis), dtype=torch.float32, device=torch_device
        )
        plane_offset = float(raw_plane_offset)
        if not math.isfinite(plane_offset):
            raise ValueError("clip plane offset must be finite")

    # Execute the exact configured iteration budget. Center and radius gradients
    # are clipped independently because their parameter scales and rates differ.
    final_loss_value = 0.0
    for _ in range(params.n_iters):
        optimizer.zero_grad(set_to_none=True)
        centers = centers_normalized * mesh_extent
        radii_normalized = (
            torch_functional.softplus(raw_radii) + minimum_radius_normalized
        )
        radii = radii_normalized * mesh_extent
        loss = (
            params.lambda_coverage
            * _coverage_loss(coverage_points, centers, radii, mesh_extent)
            + params.lambda_protrusion
            * _protrusion_loss(centers, radii, directions, mesh_query, mesh_extent)
            + params.lambda_tangency
            * _tangency_loss(centers, radii, mesh_query, mesh_extent)
            + params.lambda_overlap * _overlap_loss(centers, radii, mesh_extent)
        )
        if plane_normal is not None:
            loss = loss + params.lambda_clip_plane * _halfplane_loss(
                centers,
                radii,
                plane_normal,
                plane_offset,
                buffer=params.clip_plane_buffer,
            )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite per-link refinement loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [centers_normalized],
            max_norm=params.grad_clip_norm,
        )
        torch.nn.utils.clip_grad_norm_(
            [raw_radii],
            max_norm=params.grad_clip_norm,
        )
        optimizer.step()
        final_loss_value = float(loss.detach())

    final_centers = centers_normalized.detach() * mesh_extent
    final_radii = (
        torch_functional.softplus(raw_radii).detach() + minimum_radius_normalized
    ) * mesh_extent

    # Hard-project any residual clip-plane violations after the soft penalty converges.
    if plane_normal is not None:
        clearance = final_centers @ plane_normal - final_radii - plane_offset
        violations = torch.relu(-clearance)
        final_centers = final_centers + violations[:, None] * plane_normal[None, :]

    logger.info(
        "Refined link with {} spheres: loss {:.6g}",
        len(initial_spheres),
        final_loss_value,
    )

    # Convert detached tensors back to the public Sphere representation in input order.
    centers_numpy = final_centers.cpu().numpy()
    radii_numpy = final_radii.cpu().numpy()
    return [
        Sphere(center=centers_numpy[index], radius=np.asarray(radii_numpy[index]))
        for index in range(len(initial_spheres))
    ]


def refine_robot_spheres(
    link_spheres: dict[str, list[Sphere]],
    link_meshes: dict[str, trimesh.Trimesh],
    refine_params: RefineParams | None = None,
    clip_links: Optional[dict[str, ClipPlane]] = None,
) -> dict[str, list[Sphere]]:
    """Refine robot collision spheres independently for every link.

    Args:
        link_spheres: Initial link-local spheres keyed by link name.
        link_meshes: Original link-local collision meshes keyed by link name.
        refine_params: Optional per-link optimization parameters.
        clip_links: Optional link-local clipping plane per named link.

    Returns:
        Refined spheres keyed by link name, preserving links without spheres.

    Raises:
        ValueError: If a clip link has no sphere-bearing collision mesh.
    """
    params = refine_params or RefineParams()
    clip_configuration = clip_links or {}
    unknown_clip_links = set(clip_configuration).difference(
        link_name for link_name, spheres in link_spheres.items() if spheres
    )
    if unknown_clip_links:
        raise ValueError(
            "Clip links do not contain collision spheres: "
            + ", ".join(sorted(unknown_clip_links))
        )

    refined: dict[str, list[Sphere]] = {}
    for link_name, spheres in link_spheres.items():
        mesh = link_meshes.get(link_name)
        if not spheres or mesh is None or mesh.is_empty:
            refined[link_name] = spheres
            continue
        refined[link_name] = refine_link_spheres(
            mesh=mesh,
            initial_spheres=spheres,
            params=params,
            clip_plane=clip_configuration.get(link_name),
        )
    return refined
