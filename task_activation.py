"""task supervisor에서 상속받는 일회성 활성화 gate.

supervisor는 preload task에서 고비용 worker를 시작하고 익명 pipe의 읽기 쪽을
이 환경 변수로 전달한다. 모델 초기화가 진행 중이어도 활성화 task에서 쓴 byte는
buffer에 남으므로 느린 시작 과정에서 task 4 trigger가 유실되지 않는다.
독립 실행한 worker는 이 변수를 받지 않고 즉시 계속 진행한다.
"""

import os


ACTIVATION_FD_ENV = "FOUNDATIONPOSE_ACTIVATION_FD"


def wait_for_supervisor_activation() -> bool:
    """supervisor의 일회성 활성화 gate를 상속받았다면 기다린다.

    supervisor가 관리하는 worker이면 ``True``, 직접 실행한 worker이면
    ``False``를 반환한다.
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
