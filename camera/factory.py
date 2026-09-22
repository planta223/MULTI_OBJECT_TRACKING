"""애플리케이션 진입점에서 카메라 adapter를 지연 탐색한다."""

from importlib import import_module
import re

from .base import CameraSource


_CAMERA_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def create_camera_source(config) -> CameraSource:
    """``config.camera_type``으로 지정된 adapter를 생성한다.

    카메라 adapter 모듈은 ``camera/<camera_type>.py``에 위치하고
    ``create_source(config)`` 함수를 제공해야 한다. 이 factory를 import하는
    것만으로는 제조사 SDK나 구체적인 카메라 패키지를 불러오지 않는다.
    """

    camera_type = getattr(config, "camera_type", None)
    if not isinstance(camera_type, str) or not _CAMERA_TYPE_PATTERN.fullmatch(
        camera_type
    ):
        raise ValueError(
            "camera_type must be a lowercase Python module name; "
            f"got {camera_type!r}."
        )

    module_name = f"{__package__}.{camera_type}"
    try:
        adapter_module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise ValueError(
                f"No camera adapter module exists for {camera_type!r}: "
                f"expected {module_name}."
            ) from error
        raise

    adapter_factory = getattr(adapter_module, "create_source", None)
    if not callable(adapter_factory):
        raise TypeError(
            f"Camera adapter {module_name} must define create_source(config)."
        )

    source = adapter_factory(config)
    if not isinstance(source, CameraSource):
        raise TypeError(
            f"Camera adapter {module_name} returned {type(source).__name__}, "
            "not a CameraSource."
        )
    return source
