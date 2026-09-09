"""FoundationPose state and CAD assets for one tracked object."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional, Union

import numpy as np

from config import FoundationPoseConfig, ObjectConfig
from constants import TrackingMode, TrackingState
from .foundationpose_runtime import FoundationPoseRuntime
from camera.base import FrameData
from .pose_result import PoseResult


PathLike = Union[str, Path]
_MIN_MASK_PIXELS = 4
_MIN_VALID_DEPTH_PIXELS = 4


@dataclass(frozen=True)
class CadModel:
    """Per-object CAD data in meters plus its oriented bounding box."""

    mesh: Any
    model_path: Path
    to_origin: np.ndarray
    extents: np.ndarray
    bbox: np.ndarray


def load_cad_model(object_config: ObjectConfig) -> CadModel:
    """Load, merge, scale, and validate one CAD model with trimesh."""

    try:
        import trimesh
    except ImportError as error:
        raise ImportError(
            "trimesh is required to load CAD models. Use the FoundationPose "
            "environment or install the project dependencies."
        ) from error

    mesh_scale = float(object_config.mesh_scale_to_meter)
    if not np.isfinite(mesh_scale) or mesh_scale <= 0:
        raise ValueError(
            f"mesh_scale_to_meter must be finite and positive for "
            f"{object_config.object_id}; received {mesh_scale}."
        )

    model_path = Path(object_config.model_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"CAD file not found for {object_config.object_id}: {model_path}"
        )

    try:
        loaded = trimesh.load(str(model_path))
    except Exception as error:
        raise RuntimeError(
            f"Failed to load CAD for {object_config.object_id}: {model_path}"
        ) from error

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(
                f"CAD scene has no geometry for {object_config.object_id}: "
                f"{model_path}"
            )
        try:
            loaded = loaded.dump(concatenate=True)
        except Exception as error:
            raise ValueError(
                f"CAD scene cannot be merged for {object_config.object_id}: "
                f"{model_path}"
            ) from error

    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(
            f"CAD did not produce a Trimesh for {object_config.object_id}: "
            f"{type(loaded).__name__}"
        )

    mesh = loaded.copy()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(
            f"CAD mesh is empty for {object_config.object_id}: {model_path}"
        )

    mesh.apply_scale(mesh_scale)
    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError(
            f"CAD contains non-finite vertices for {object_config.object_id}: "
            f"{model_path}"
        )

    normals = np.asarray(mesh.vertex_normals)
    if normals.shape != mesh.vertices.shape or not np.all(np.isfinite(normals)):
        raise ValueError(
            f"CAD vertex normals are invalid for {object_config.object_id}: "
            f"{model_path}"
        )

    try:
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    except Exception as error:
        raise ValueError(
            f"Failed to compute oriented bounds for {object_config.object_id}: "
            f"{model_path}"
        ) from error

    to_origin = np.asarray(to_origin, dtype=np.float64).reshape(4, 4)
    extents = np.asarray(extents, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(to_origin)) or not np.all(np.isfinite(extents)):
        raise ValueError(
            f"CAD oriented bounds are non-finite for {object_config.object_id}."
        )
    if np.any(extents <= 0):
        raise ValueError(
            f"CAD oriented bounds must have positive extents for "
            f"{object_config.object_id}: {extents}"
        )

    bbox = np.stack((-extents / 2.0, extents / 2.0), axis=0)
    return CadModel(
        mesh=mesh,
        model_path=model_path,
        to_origin=to_origin,
        extents=extents,
        bbox=bbox,
    )


class FoundationPoseTracker:
    """Manage one object-specific FoundationPose estimator and its pose_last state."""

    def __init__(
        self,
        object_config: ObjectConfig,
        foundationpose_config: FoundationPoseConfig,
        runtime: FoundationPoseRuntime,
        debug_dir: PathLike,
    ) -> None:
        self.object_config = object_config
        self.foundationpose_config = foundationpose_config
        self.runtime = runtime
        self.cad = load_cad_model(object_config)
        self.mesh = self.cad.mesh
        self.to_origin = self.cad.to_origin
        self.extents = self.cad.extents
        self.bbox = self.cad.bbox
        self.estimator = runtime.create_estimator(
            mesh=self.mesh,
            debug=foundationpose_config.debug,
            debug_dir=debug_dir,
        )
        self.state = TrackingState.UNINITIALIZED
        self.last_result: Optional[PoseResult] = None

    @property
    def object_id(self) -> str:
        return self.object_config.object_id

    @staticmethod
    def _foundationpose_input(array: np.ndarray) -> np.ndarray:
        """Return a writable C-contiguous view/copy for PyTorch interop.

        FrameData owns read-only snapshots by design.  FoundationPose passes
        its NumPy inputs to ``torch.as_tensor()``, which requires writable
        arrays for supported behavior.  ``np.require`` reuses an array that
        already satisfies both requirements and copies only when necessary.
        """

        return np.require(array, requirements=("C", "W"))

    def validate_registration_mask(
        self,
        frame: FrameData,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Normalize a numeric binary mask and validate usable depth support."""

        if not isinstance(mask, np.ndarray):
            raise TypeError(f"Mask for {self.object_id} must be a NumPy array.")
        if mask.ndim != 2 or mask.shape != (frame.height, frame.width):
            raise ValueError(
                f"Mask resolution mismatch for {self.object_id}: expected "
                f"{(frame.height, frame.width)}, received {mask.shape}."
            )
        if mask.dtype != np.bool_ and not np.issubdtype(mask.dtype, np.number):
            raise TypeError(
                f"Mask for {self.object_id} must be bool or numeric; "
                f"received {mask.dtype}."
            )
        if np.issubdtype(mask.dtype, np.floating) and not np.all(np.isfinite(mask)):
            raise ValueError(f"Mask for {self.object_id} contains non-finite values.")

        normalized = np.ascontiguousarray(mask != 0, dtype=np.bool_)
        mask_pixels = int(np.count_nonzero(normalized))
        if mask_pixels < _MIN_MASK_PIXELS:
            raise ValueError(
                f"Mask for {self.object_id} has only {mask_pixels} foreground "
                f"pixels; at least {_MIN_MASK_PIXELS} are required."
            )

        valid_depth = (
            normalized & np.isfinite(frame.depth_m) & (frame.depth_m >= 0.001)
        )
        valid_depth_pixels = int(np.count_nonzero(valid_depth))
        if valid_depth_pixels < _MIN_VALID_DEPTH_PIXELS:
            raise ValueError(
                f"Mask for {self.object_id} has only {valid_depth_pixels} valid "
                "depth pixels; at least "
                f"{_MIN_VALID_DEPTH_PIXELS} are required."
            )
        return normalized

    @staticmethod
    def _validate_pose(pose: Any) -> np.ndarray:
        pose_array = np.asarray(pose)
        if pose_array.shape != (4, 4):
            raise ValueError(
                f"FoundationPose returned pose shape {pose_array.shape}, not (4, 4)."
            )
        if not np.issubdtype(pose_array.dtype, np.number):
            raise TypeError(
                f"FoundationPose returned non-numeric pose dtype {pose_array.dtype}."
            )
        if not np.all(np.isfinite(pose_array)):
            raise ValueError("FoundationPose returned a non-finite pose matrix.")
        return np.array(pose_array, copy=True, order="C")

    def _failure_result(
        self,
        frame: FrameData,
        cycle_id: int,
        mode: TrackingMode,
        processing_time_s: float,
        error: Exception,
        failure_state: TrackingState,
    ) -> PoseResult:
        self.state = failure_state
        result = PoseResult(
            cycle_id=cycle_id,
            object_id=self.object_id,
            source_frame_id=frame.source_frame_id,
            host_wall_time_s=frame.host_wall_time_s,
            host_monotonic_time_s=frame.host_monotonic_time_s,
            mode=mode,
            state=self.state,
            valid=False,
            pose=None,
            processing_time_s=processing_time_s,
            message=f"{type(error).__name__}: {error}",
        )
        self.last_result = result
        return result

    def register(
        self,
        frame: FrameData,
        mask: np.ndarray,
        cycle_id: int,
    ) -> PoseResult:
        normalized_mask = self.validate_registration_mask(frame, mask)
        previous_state = self.state
        start_time = time.perf_counter()
        try:
            rgb = self._foundationpose_input(frame.rgb)
            depth_m = self._foundationpose_input(frame.depth_m)
            K = self._foundationpose_input(frame.K)
            with self.runtime.inference_guard():
                pose = self.estimator.register(
                    K=K,
                    rgb=rgb,
                    depth=depth_m,
                    ob_mask=normalized_mask,
                    iteration=self.foundationpose_config.register_refine_iter,
                )
            if self.estimator.pose_last is None:
                raise RuntimeError(
                    "FoundationPose register() did not initialize tracking state."
                )
            pose_array = self._validate_pose(pose)
        except Exception as error:
            elapsed = time.perf_counter() - start_time
            failure_state = (
                TrackingState.UNINITIALIZED
                if previous_state is TrackingState.UNINITIALIZED
                else TrackingState.LOST
            )
            return self._failure_result(
                frame,
                cycle_id,
                TrackingMode.REGISTER,
                elapsed,
                error,
                failure_state,
            )

        elapsed = time.perf_counter() - start_time
        self.state = TrackingState.TRACKING
        result = PoseResult(
            cycle_id=cycle_id,
            object_id=self.object_id,
            source_frame_id=frame.source_frame_id,
            host_wall_time_s=frame.host_wall_time_s,
            host_monotonic_time_s=frame.host_monotonic_time_s,
            mode=TrackingMode.REGISTER,
            state=self.state,
            valid=True,
            pose=pose_array,
            processing_time_s=elapsed,
        )
        self.last_result = result
        return result

    def track(self, frame: FrameData, cycle_id: int) -> PoseResult:
        if self.state is not TrackingState.TRACKING:
            raise RuntimeError(
                f"Cannot track {self.object_id} while state={self.state.value}; "
                "register() must succeed first."
            )

        start_time = time.perf_counter()
        try:
            rgb = self._foundationpose_input(frame.rgb)
            depth_m = self._foundationpose_input(frame.depth_m)
            K = self._foundationpose_input(frame.K)
            with self.runtime.inference_guard():
                pose = self.estimator.track_one(
                    rgb=rgb,
                    depth=depth_m,
                    K=K,
                    iteration=self.foundationpose_config.track_refine_iter,
                )
            pose_array = self._validate_pose(pose)
        except Exception as error:
            elapsed = time.perf_counter() - start_time
            return self._failure_result(
                frame,
                cycle_id,
                TrackingMode.TRACK,
                elapsed,
                error,
                TrackingState.LOST,
            )

        elapsed = time.perf_counter() - start_time
        result = PoseResult(
            cycle_id=cycle_id,
            object_id=self.object_id,
            source_frame_id=frame.source_frame_id,
            host_wall_time_s=frame.host_wall_time_s,
            host_monotonic_time_s=frame.host_monotonic_time_s,
            mode=TrackingMode.TRACK,
            state=self.state,
            valid=True,
            pose=pose_array,
            processing_time_s=elapsed,
        )
        self.last_result = result
        return result

# Compatibility name retained for the proven tracking layer contract.
ObjectTracker = FoundationPoseTracker

