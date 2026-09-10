"""Lazy camera-adapter discovery for application entrypoints."""

from importlib import import_module
import re

from .base import CameraSource


_CAMERA_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def create_camera_source(config) -> CameraSource:
    """Create the adapter named by ``config.camera_type``.

    A camera adapter module must live at ``camera/<camera_type>.py`` and expose
    a ``create_source(config)`` function. Importing this factory does not load
    any vendor SDK or concrete camera package.
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
