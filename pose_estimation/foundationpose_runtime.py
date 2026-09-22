"""공유하는 고비용 FoundationPose 추론 리소스.

NVIDIA FoundationPose 모듈을 직접 가져오는 유일한 애플리케이션 모듈이다.
GPU container 밖에서도 패키지의 데이터 계약을 불러올 수 있도록 import와
초기화를 의도적으로 지연한다.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Optional, Union


PathLike = Union[str, Path]


class FoundationPoseRuntime:
    """scorer, refiner, CUDA raster context를 각각 하나씩 소유한다.

    여러 ObjectTracker 인스턴스가 이 runtime을 통해 독립적인 FoundationPose
    estimator를 생성할 수 있다. 공유 추론 리소스를 사용하는 호출은
    ``inference_guard``가 순차 실행한다.
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

        # 외부 checkout은 기본적으로 애플리케이션 루트 안에 위치한다.
        return (Path(__file__).resolve().parents[1] / "FoundationPose").resolve()

    def initialize(self) -> None:
        """FoundationPose를 가져오고 공유 리소스를 정확히 한 번 할당한다."""

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
            # NVIDIA FoundationPose 직접 import는 모두 이 모듈 안에 둔다.
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
        """독립적인 mesh와 tracking 상태를 가진 estimator를 생성한다."""

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
        """공유 predictor/context 객체를 사용하는 호출을 순차 실행한다."""

        with self._inference_lock:
            yield
