"""Shared heavyweight FoundationPose inference resources.

This is the only application module that imports NVIDIA FoundationPose
modules directly. Import and initialization are deliberately lazy so the
package's data contracts remain importable outside the GPU container.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Optional, Union


PathLike = Union[str, Path]


class FoundationPoseRuntime:
    """Own one scorer, one refiner, and one CUDA raster context.

    Multiple ObjectTracker instances may create independent FoundationPose
    estimators through this runtime. Calls using the shared inference resources
    are serialized by ``inference_guard``.
    """

    def __init__(self, foundationpose_root: Optional[PathLike] = None) -> None:
        self.foundationpose_root = self._resolve_root(foundationpose_root)
        self._inference_lock = threading.RLock()
        self._initialized = False
        self._foundationpose_class: Optional[Any] = None
        self._scorer: Optional[Any] = None
        self._refiner: Optional[Any] = None
        self._glctx: Optional[Any] = None
        self.initialize()

    @staticmethod
    def _resolve_root(configured_root: Optional[PathLike]) -> Path:
        if configured_root is not None:
            return Path(configured_root).expanduser().resolve()

        environment_root = os.environ.get("FOUNDATIONPOSE_ROOT")
        if environment_root:
            return Path(environment_root).expanduser().resolve()

        # The external checkout lives inside the application root by default.
        return (Path(__file__).resolve().parents[1] / "FoundationPose").resolve()

    def initialize(self) -> None:
        """Import FoundationPose and allocate shared resources exactly once."""

        if self._initialized:
            return

        estimater_path = self.foundationpose_root / "estimater.py"
        if not estimater_path.is_file():
            raise FileNotFoundError(
                f"FoundationPose estimater.py not found under "
                f"{self.foundationpose_root}. Install the external dependency with:\n"
                f"  cd {self.foundationpose_root.parent}\n"
                "  git clone https://github.com/NVlabs/FoundationPose.git FoundationPose"
            )

        root_text = str(self.foundationpose_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        existing_estimater = sys.modules.get("estimater")
        if existing_estimater is not None:
            existing_file = Path(existing_estimater.__file__).resolve()
            if existing_file != estimater_path.resolve():
                raise RuntimeError(
                    "A different estimater module is already imported: "
                    f"{existing_file}"
                )

        try:
            # Keep all direct NVIDIA FoundationPose imports in this module.
            from estimater import FoundationPose
            from learning.training.predict_pose_refine import PoseRefinePredictor
            from learning.training.predict_score import ScorePredictor
            import nvdiffrast.torch as dr
        except Exception as error:
            raise ImportError(
                "Failed to import FoundationPose dependencies. Run inside the "
                "FoundationPose GPU environment and ensure FOUNDATIONPOSE_ROOT "
                f"points to a valid checkout (current: {self.foundationpose_root}). Clone NVlabs/FoundationPose there or pass --foundationpose-root."
            ) from error

        self._foundationpose_class = FoundationPose
        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()
        self._glctx = dr.RasterizeCudaContext()
        self._initialized = True

    @property
    def scorer(self) -> Any:
        if self._scorer is None:
            raise RuntimeError("FoundationPoseRuntime is not initialized.")
        return self._scorer

    @property
    def refiner(self) -> Any:
        if self._refiner is None:
            raise RuntimeError("FoundationPoseRuntime is not initialized.")
        return self._refiner

    @property
    def glctx(self) -> Any:
        if self._glctx is None:
            raise RuntimeError("FoundationPoseRuntime is not initialized.")
        return self._glctx

    def create_estimator(
        self,
        mesh: Any,
        debug: int,
        debug_dir: PathLike,
    ) -> Any:
        """Create an estimator with independent mesh and tracking state."""

        if self._foundationpose_class is None:
            raise RuntimeError("FoundationPoseRuntime is not initialized.")

        debug_path = Path(debug_dir).expanduser()
        debug_path.mkdir(parents=True, exist_ok=True)
        return self._foundationpose_class(
            model_pts=mesh.vertices.copy(),
            model_normals=mesh.vertex_normals.copy(),
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
            debug=debug,
            debug_dir=str(debug_path),
        )

    @contextmanager
    def inference_guard(self) -> Iterator[None]:
        """Serialize calls that use the shared predictor/context objects."""

        with self._inference_lock:
            yield
