"""One-shot activation gate inherited from the task supervisor.

The supervisor starts the heavyweight worker during the preload task and passes
the read side of an anonymous pipe through this environment variable.  A byte
written at the activation task remains buffered if model initialization is
still in progress, so the task-4 trigger cannot be lost during a slow startup.
Standalone workers do not receive the variable and continue immediately.
"""

import os


ACTIVATION_FD_ENV = "FOUNDATIONPOSE_ACTIVATION_FD"


def wait_for_supervisor_activation() -> bool:
    """Block for the supervisor's one-shot activation, if one was inherited.

    Returns ``True`` for a supervisor-managed worker and ``False`` when the
    worker was launched directly.
    """

    raw_fd = os.environ.pop(ACTIVATION_FD_ENV, None)
    if raw_fd is None:
        return False

    print(
        "FoundationPose preload complete; waiting for task activation.",
        flush=True,
    )

    try:
        fd = int(raw_fd)
    except ValueError as error:
        raise RuntimeError(
            f"Invalid inherited activation descriptor: {raw_fd!r}."
        ) from error
    if fd < 0:
        raise RuntimeError(f"Invalid inherited activation descriptor: {fd}.")

    try:
        signal_byte = os.read(fd, 1)
    except OSError as error:
        raise RuntimeError("Failed while waiting for task activation.") from error
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    if not signal_byte:
        raise RuntimeError("Task supervisor closed before activating the worker.")
    return True
