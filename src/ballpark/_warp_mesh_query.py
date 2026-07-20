"""Warp-backed differentiable signed-distance queries for triangle meshes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import trimesh
import warp as wp


@wp.kernel
def _query_sdf_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    output_sdf: wp.array(dtype=wp.float32),
    output_gradient: wp.array(dtype=wp.vec3),
    max_distance: wp.float32,
) -> None:
    """Compute signed distance and its spatial gradient for query points.

    Args:
        mesh_id: Warp identifier for the triangle mesh queried by the kernel.
        query_points: Link-local query points with shape ``(N,)`` as vectors
            measured in meters.
        output_sdf: Preallocated signed-distance output array with shape ``(N,)``.
        output_gradient: Preallocated spatial-gradient output array with shape
            ``(N,)`` as vectors.
        max_distance: Maximum mesh-query distance in meters and fallback SDF
            value for points with no query result.

    Note:
        The kernel writes results into ``output_sdf`` and ``output_gradient``;
        Warp kernels do not return Python values.
    """
    point_index = wp.tid()
    point = query_points[point_index]
    query = wp.mesh_query_point(mesh_id, point, max_distance)
    if not query.result:
        output_sdf[point_index] = max_distance
        output_gradient[point_index] = wp.vec3(0.0, 0.0, 0.0)
        return

    closest_point = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    offset = point - closest_point
    distance = wp.length(offset)
    signed_distance = distance * query.sign
    gradient = wp.vec3(0.0, 0.0, 0.0)
    if distance > 1.0e-8:
        gradient = offset / distance * query.sign
    output_sdf[point_index] = signed_distance
    output_gradient[point_index] = gradient


class WarpMeshQuery:
    """Reusable Warp BVH providing signed-distance queries for one mesh.

    Args:
        mesh: Original link-local collision mesh in meters.
        device: Torch device used for query tensors. CPU and CUDA devices use
            the same real Warp kernel implementation.
    """

    def __init__(self, mesh: trimesh.Trimesh, device: torch.device) -> None:
        """Build a reusable Warp mesh from the source triangle mesh.

        Args:
            mesh: Original link-local collision mesh in meters.
            device: Torch device used for Warp/Torch interoperability.

        Raises:
            ValueError: If the source mesh contains no triangles.
        """
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            raise ValueError("Warp mesh queries require a non-empty triangle mesh")

        wp.init()
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self._warp_device = wp.device_from_torch(device)
        self._vertices = wp.array(vertices, dtype=wp.vec3, device=self._warp_device)
        self._faces = wp.array(faces.ravel(), dtype=wp.int32, device=self._warp_device)
        self._mesh = wp.Mesh(points=self._vertices, indices=self._faces)
        mesh_extent = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        self._max_distance = max(mesh_extent + 0.01, 0.01)

    def query_sdf(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Query signed distance and analytic gradient for link-local points.

        Args:
            points: Contiguous or non-contiguous tensor of shape ``(N, 3)`` in
                meters on the configured Torch device.

        Returns:
            Signed distances ``(N,)`` and spatial gradients ``(N, 3)``. SDF is
            negative inside the mesh and positive outside.

        Raises:
            ValueError: If points do not have shape ``(N, 3)``.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("query points must have shape (N, 3)")
        contiguous_points = points.to(device=self.device, dtype=torch.float32).contiguous()
        signed_distance = torch.empty(
            contiguous_points.shape[0], dtype=torch.float32, device=self.device
        )
        sdf_gradient = torch.empty_like(contiguous_points)
        wp.launch(
            kernel=_query_sdf_kernel,
            dim=contiguous_points.shape[0],
            inputs=[
                self._mesh.id,
                wp.from_torch(contiguous_points, dtype=wp.vec3),
                wp.from_torch(signed_distance),
                wp.from_torch(sdf_gradient, dtype=wp.vec3),
                self._max_distance,
            ],
            device=self._warp_device,
        )
        wp.synchronize_device(self._warp_device)
        return signed_distance, sdf_gradient


class WarpSignedDistance(torch.autograd.Function):
    """Torch autograd seam for Warp signed-distance queries."""

    @staticmethod
    def forward(
        context: Any,
        points: torch.Tensor,
        mesh_query: WarpMeshQuery,
    ) -> torch.Tensor:
        """Compute SDF values and retain analytic point gradients.

        Args:
            context: Torch autograd context.
            points: Query points of shape ``(N, 3)`` in meters.
            mesh_query: Reusable Warp mesh query instance.

        Returns:
            Signed-distance tensor of shape ``(N,)``.
        """
        signed_distance, sdf_gradient = mesh_query.query_sdf(points.detach())
        context.save_for_backward(sdf_gradient)
        return signed_distance

    @staticmethod
    def backward(
        context: Any,
        output_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Propagate SDF gradients to the input point tensor.

        Args:
            context: Torch autograd context containing analytic SDF gradients.
            output_gradient: Upstream gradient of shape ``(N,)``.

        Returns:
            Point gradient of shape ``(N, 3)`` and ``None`` for mesh query.
        """
        (sdf_gradient,) = context.saved_tensors
        return output_gradient.unsqueeze(-1) * sdf_gradient, None
