"""Exact mesh collision analysis across low-discrepancy joint samples."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import fcl
import numpy as np
import trimesh
from loguru import logger
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


CanonicalPair = tuple[str, str]


def canonicalize_link_pair(link_a: str, link_b: str) -> CanonicalPair:
    """Return a link pair in deterministic lexicographic order.

    Args:
        link_a: First link name.
        link_b: Second link name.

    Returns:
        Canonically ordered pair of link names.
    """
    return tuple(sorted((link_a, link_b)))


def _build_collision_object(mesh: trimesh.Trimesh) -> fcl.CollisionObject:
    """Build one reusable FCL object from a link-local triangle mesh.

    Args:
        mesh: Link-local collision mesh in meters.

    Returns:
        FCL collision object backed by a persistent BVH model.

    Raises:
        ValueError: If the collision mesh contains no triangles.
    """
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError("FCL collision meshes must contain triangles")

    geometry = fcl.BVHModel()
    geometry.beginModel(int(vertices.shape[0]), int(faces.shape[0]))
    geometry.addSubModel(vertices, faces)
    geometry.endModel()
    return fcl.CollisionObject(geometry)


def _build_candidate_collision_objects(
    link_meshes: dict[str, trimesh.Trimesh],
    candidate_pairs: set[CanonicalPair],
) -> dict[str, fcl.CollisionObject]:
    """Build reusable FCL objects only for links in candidate pairs.

    Args:
        link_meshes: Link-local collision meshes keyed by link name.
        candidate_pairs: Canonical link pairs that may be checked.

    Returns:
        FCL collision objects keyed by candidate link name.

    Raises:
        KeyError: If a candidate link has no collision mesh.
        ValueError: If a candidate collision mesh contains no triangles.
    """
    candidate_link_names = {link_name for pair in candidate_pairs for link_name in pair}
    return {
        link_name: _build_collision_object(link_meshes[link_name])
        for link_name in candidate_link_names
    }


def _compute_validated_transforms(
    joint_configuration: np.ndarray,
    compute_transforms: Callable[[np.ndarray], np.ndarray],
    all_link_names: list[str],
) -> np.ndarray:
    """Compute and validate link transforms for one joint configuration.

    Args:
        joint_configuration: Joint positions with shape ``(D,)`` in
            radians/meters.
        compute_transforms: FK callback returning world link poses as
            ``[qw, qx, qy, qz, x, y, z]``.
        all_link_names: Link ordering corresponding to FK output rows.

    Returns:
        Link transforms with shape ``(L, 7)``.

    Raises:
        RuntimeError: If FK output has an incompatible shape.
    """
    transforms = np.asarray(
        compute_transforms(np.asarray(joint_configuration, dtype=np.float64)),
        dtype=np.float64,
    )
    expected_shape = (len(all_link_names), 7)
    if transforms.shape != expected_shape:
        raise RuntimeError(
            f"FK must return shape {expected_shape}, got {transforms.shape}"
        )
    return transforms


def _transform_key(collision_object: fcl.CollisionObject) -> tuple[float, ...]:
    """Create a stable lookup key from an FCL world transform.

    Args:
        collision_object: FCL object whose transform identifies its link.

    Returns:
        Rounded rotation and translation values suitable for lookup across
        python-fcl callback wrappers.
    """
    rotation = np.asarray(collision_object.getRotation(), dtype=np.float64)
    translation = np.asarray(collision_object.getTranslation(), dtype=np.float64)
    return tuple(np.round(np.concatenate((rotation.ravel(), translation)), 12))


def _update_collision_object_transforms(
    collision_objects: dict[str, fcl.CollisionObject],
    transforms: np.ndarray,
    link_indices: dict[str, int],
) -> dict[tuple[float, ...], set[str]]:
    """Apply FK world poses to reusable FCL collision objects.

    Args:
        collision_objects: Persistent FCL objects keyed by link name.
        transforms: Validated link poses with shape ``(L, 7)`` encoded as
            ``[qw, qx, qy, qz, x, y, z]`` in world coordinates and meters.
        link_indices: Mapping from link name to FK output row.

    Returns:
        World-transform keys mapped to all co-located link names. This mapping
        preserves conservative pair attribution through python-fcl callbacks.
    """
    transform_names: dict[tuple[float, ...], set[str]] = {}
    for link_name, collision_object in collision_objects.items():
        transform = transforms[link_indices[link_name]]
        # SciPy consumes xyzw while Ballpark's FK seam returns wxyz.
        rotation_xyzw = np.roll(transform[:4], -1)
        rotation_matrix = Rotation.from_quat(rotation_xyzw).as_matrix()
        collision_object.setTransform(fcl.Transform(rotation_matrix, transform[4:]))
        transform_names.setdefault(_transform_key(collision_object), set()).add(
            link_name
        )
    return transform_names


def _objects_collide(
    object_a: fcl.CollisionObject,
    object_b: fcl.CollisionObject,
    collision_request: fcl.CollisionRequest,
) -> bool:
    """Check exact triangle-mesh collision for two positioned FCL objects.

    Args:
        object_a: First positioned FCL collision object.
        object_b: Second positioned FCL collision object.
        collision_request: Shared exact-collision request configuration.

    Returns:
        Whether the two meshes collide at their current world transforms.
    """
    collision_result = fcl.CollisionResult()
    return fcl.collide(object_a, object_b, collision_request, collision_result) > 0


def compute_configuration_collision_pairs(
    link_meshes: dict[str, trimesh.Trimesh],
    candidate_pairs: set[CanonicalPair],
    joint_configuration: np.ndarray,
    compute_transforms: Callable[[np.ndarray], np.ndarray],
    all_link_names: list[str],
) -> set[CanonicalPair]:
    """Find exact mesh collisions at one robot joint configuration.

    Args:
        link_meshes: Link-local collision meshes keyed by link name.
        candidate_pairs: Canonical non-adjacent link pairs to analyze.
        joint_configuration: Joint positions with shape ``(D,)`` in
            radians/meters.
        compute_transforms: FK callback returning ``(L, 7)`` rows encoded as
            ``[qw, qx, qy, qz, x, y, z]`` in world coordinates and meters.
        all_link_names: Link order corresponding to FK output rows.

    Returns:
        Canonical candidate pairs colliding at the supplied configuration.

    Raises:
        RuntimeError: If FK output has an incompatible shape.
        ValueError: If a collision mesh contains no triangles.
    """
    canonical_candidates = {
        canonicalize_link_pair(link_a, link_b) for link_a, link_b in candidate_pairs
    }
    if not canonical_candidates:
        return set()

    # Use the shared FK/FCL preparation path so single-configuration semantics
    # cannot drift from the sampled collision path.
    transforms = _compute_validated_transforms(
        joint_configuration,
        compute_transforms,
        all_link_names,
    )
    link_indices = {name: index for index, name in enumerate(all_link_names)}
    collision_objects = _build_candidate_collision_objects(
        link_meshes,
        canonical_candidates,
    )
    _update_collision_object_transforms(
        collision_objects,
        transforms,
        link_indices,
    )

    # Direct pair iteration is cheaper than broadphase for a single pose, but
    # exact collision observation still uses the same shared FCL kernel.
    collision_request = fcl.CollisionRequest(num_max_contacts=1)
    observed_collisions: set[CanonicalPair] = set()
    for link_a, link_b in canonical_candidates:
        if _objects_collide(
            collision_objects[link_a],
            collision_objects[link_b],
            collision_request,
        ):
            observed_collisions.add((link_a, link_b))
    return observed_collisions


def compute_halton_ignore_pairs(
    link_meshes: dict[str, trimesh.Trimesh],
    candidate_pairs: set[CanonicalPair],
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    compute_transforms: Callable[[np.ndarray], np.ndarray],
    all_link_names: list[str],
    n_samples: int,
    seed: int,
) -> set[CanonicalPair]:
    """Find candidate link pairs never colliding in Halton joint samples.

    The function creates one FCL BVH and collision object per link, then only
    updates world transforms for each sampled joint configuration. Broadphase
    callbacks invoke exact triangle-mesh collision checks and accumulate every
    candidate pair observed in collision.

    Args:
        link_meshes: Link-local collision meshes keyed by link name.
        candidate_pairs: Canonical non-adjacent link pairs to analyze.
        joint_lower: Lower joint limits with shape ``(D,)`` in radians/meters.
        joint_upper: Upper joint limits with shape ``(D,)`` in radians/meters.
        compute_transforms: FK callback returning ``(L, 7)`` rows encoded as
            ``[qw, qx, qy, qz, x, y, z]`` in world coordinates and meters.
        all_link_names: Link order corresponding to FK output rows.
        n_samples: Positive number of Halton configurations to evaluate.
        seed: Scrambling seed used by SciPy's Halton generator.

    Returns:
        Canonical candidate pairs with no collision observed in the samples.

    Raises:
        ValueError: If sample count or joint limits are invalid.
        RuntimeError: If FK output has an incompatible shape.
    """
    # Validate the sampling request and convert joint bounds to one-dimensional
    # finite arrays before constructing any collision state.
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("joint limits must be matching one-dimensional arrays")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("joint limits must be finite")
    if np.any(lower >= upper):
        raise ValueError("every joint lower limit must be smaller than its upper limit")

    canonical_candidates = {
        canonicalize_link_pair(link_a, link_b) for link_a, link_b in candidate_pairs
    }
    if not canonical_candidates:
        return set()

    # Build one persistent BVH-backed FCL object per link so each sample only
    # needs to update world transforms rather than rebuild mesh geometry.
    collision_objects = _build_candidate_collision_objects(
        link_meshes,
        canonical_candidates,
    )
    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(list(collision_objects.values()))
    manager.setup()

    # Aggregate candidate pairs confirmed by exact collision checks across all
    # broadphase callbacks and sampled configurations.
    observed_collisions: set[CanonicalPair] = set()
    collision_request = fcl.CollisionRequest(num_max_contacts=1)
    transform_names: dict[tuple[float, ...], set[str]] = {}

    def collision_callback(
        object_a: fcl.CollisionObject,
        object_b: fcl.CollisionObject,
        _collision_data: object,
    ) -> bool:
        """Record exact collisions produced by one broadphase callback.

        Args:
            object_a: First broadphase collision object.
            object_b: Second broadphase collision object.
            _collision_data: Unused callback payload required by python-fcl.

        Returns:
            ``False`` so broadphase continues visiting candidate objects.
        """
        names_a = transform_names.get(_transform_key(object_a), set())
        names_b = transform_names.get(_transform_key(object_b), set())
        possible_pairs = {
            canonicalize_link_pair(link_a, link_b)
            for link_a in names_a
            for link_b in names_b
            if link_a != link_b
        }.intersection(canonical_candidates)
        if not possible_pairs or possible_pairs.issubset(observed_collisions):
            return False
        if _objects_collide(
            object_a,
            object_b,
            collision_request,
        ):
            # Co-located links cannot be distinguished through python-fcl's
            # callback wrappers. Conservatively mark every candidate sharing
            # both transforms to avoid producing an unsafe ignore pair.
            observed_collisions.update(possible_pairs)
        return False

    # Generate deterministic low-discrepancy joint configurations within the
    # validated limits, then evaluate FK for each complete configuration.
    halton_unit_samples = qmc.Halton(
        d=lower.size,
        scramble=True,
        seed=seed,
    ).random(n=n_samples)
    joint_samples = qmc.scale(halton_unit_samples, lower, upper)
    link_indices = {name: index for index, name in enumerate(all_link_names)}
    start_time = perf_counter()

    for sample_index, joint_cfg in enumerate(joint_samples):
        transforms = _compute_validated_transforms(
            joint_cfg,
            compute_transforms,
            all_link_names,
        )
        transform_names = _update_collision_object_transforms(
            collision_objects,
            transforms,
            link_indices,
        )

        # Refresh the broadphase from the FK-derived link poses and run exact
        # mesh checks through the callback for candidate pair aggregation.
        manager.update()
        manager.collide(None, collision_callback)

        completed_samples = sample_index + 1
        if (
            completed_samples == n_samples
            or completed_samples % max(1, n_samples // 10) == 0
        ):
            logger.info(
                "FCL/Halton: {}/{} samples, {} colliding pairs observed",
                completed_samples,
                n_samples,
                len(observed_collisions),
            )

    logger.info(
        "FCL/Halton completed {} samples in {:.3f}s",
        n_samples,
        perf_counter() - start_time,
    )
    return canonical_candidates.difference(observed_collisions)
