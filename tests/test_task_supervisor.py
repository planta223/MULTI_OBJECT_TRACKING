"""CPU-only tests for task-triggered FoundationPose worker lifecycle."""

from pathlib import Path
import os
import sys

import pytest

from scripts.run_task_supervisor import (
    TaskTriggeredWorker,
    WorkerProcessController,
    parse_args,
)
from task_activation import ACTIVATION_FD_ENV, wait_for_supervisor_activation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeController:
    def __init__(self) -> None:
        self.running = False
        self.start_calls = 0
        self.stop_calls = 0
        self.activate_calls = 0
        self.activation_available = False
        self.pid = None

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self) -> bool:
        if self.running:
            return False
        self.running = True
        self.start_calls += 1
        self.activation_available = True
        self.pid = 1234
        return True

    def stop(self) -> bool:
        if not self.running:
            return False
        self.running = False
        self.stop_calls += 1
        self.pid = None
        self.activation_available = False
        return True

    def activate(self) -> bool:
        if not self.running or not self.activation_available:
            return False
        self.activate_calls += 1
        self.activation_available = False
        return True

    def refresh(self):
        return None


class FailingController(FakeController):
    def start(self) -> bool:
        raise FileNotFoundError("missing worker interpreter")


def make_supervisor(controller, now):
    return TaskTriggeredWorker(
        controller=controller,
        preload_task_id=3,
        start_task_id=4,
        stop_task_id=6,
        task_timeout_sec=2.0,
        clock=lambda: now[0],
        log=lambda message: None,
    )


def test_worker_preloads_at_three_activates_at_four_and_stops_at_six() -> None:
    controller = FakeController()
    now = [0.0]
    supervisor = make_supervisor(controller, now)

    for task_id in (0, 1, 2):
        supervisor.on_task(task_id)
    assert controller.start_calls == 0

    supervisor.on_task(3)
    supervisor.on_task(3)
    assert controller.start_calls == 1
    assert controller.activate_calls == 0

    supervisor.on_task(4)
    supervisor.on_task(4)
    supervisor.on_task(5)
    assert controller.start_calls == 1
    assert controller.activate_calls == 1
    assert controller.stop_calls == 0
    assert controller.is_running

    supervisor.on_task(6)
    supervisor.on_task(6)
    assert controller.stop_calls == 1
    assert not controller.is_running


def test_starting_during_task_five_does_not_skip_registration_trigger() -> None:
    controller = FakeController()
    supervisor = make_supervisor(controller, [0.0])

    supervisor.on_task(5)

    assert controller.start_calls == 0


def test_task_four_cold_starts_and_activates_if_preload_was_missed() -> None:
    controller = FakeController()
    supervisor = make_supervisor(controller, [0.0])

    supervisor.on_task(4)

    assert controller.start_calls == 1
    assert controller.activate_calls == 1


def test_worker_start_failure_is_logged_without_escaping_callback() -> None:
    controller = FailingController()
    messages = []
    supervisor = TaskTriggeredWorker(
        controller=controller,
        preload_task_id=3,
        start_task_id=4,
        stop_task_id=6,
        task_timeout_sec=2.0,
        clock=lambda: 0.0,
        log=messages.append,
    )

    supervisor.on_task(3)

    assert any("FileNotFoundError" in message for message in messages)
    assert not controller.is_running


def test_later_task_stops_worker_if_task_six_message_was_missed() -> None:
    controller = FakeController()
    supervisor = make_supervisor(controller, [0.0])

    supervisor.on_task(3)
    supervisor.on_task(4)
    supervisor.on_task(7)

    assert controller.start_calls == 1
    assert controller.stop_calls == 1


def test_task_timeout_stops_worker_and_allows_task_four_to_restart() -> None:
    controller = FakeController()
    now = [10.0]
    supervisor = make_supervisor(controller, now)

    supervisor.on_task(4)
    now[0] = 12.1
    supervisor.tick()
    assert controller.stop_calls == 1

    supervisor.on_task(4)
    assert controller.start_calls == 2
    assert controller.activate_calls == 2


def test_cli_requires_worker_command_and_preserves_worker_arguments() -> None:
    with pytest.raises(SystemExit):
        parse_args([])

    args = parse_args(
        [
            "--task-topic",
            "/tasks",
            "--",
            "/opt/conda/envs/my/bin/python",
            "scripts/test_multi_object.py",
            "--camera-type",
            "ros_zed",
        ]
    )
    assert args.task_topic == "/tasks"
    assert args.preload_task_id == 3
    assert args.start_task_id == 4
    assert args.worker_cwd == PROJECT_ROOT
    assert args.worker_command[-2:] == ["--camera-type", "ros_zed"]


def test_activation_gate_is_optional_for_direct_worker(monkeypatch) -> None:
    monkeypatch.delenv(ACTIVATION_FD_ENV, raising=False)

    assert not wait_for_supervisor_activation()


def test_activation_gate_consumes_buffered_supervisor_signal(monkeypatch) -> None:
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(ACTIVATION_FD_ENV, str(read_fd))
    os.write(write_fd, b"1")
    os.close(write_fd)

    assert wait_for_supervisor_activation()
    assert ACTIVATION_FD_ENV not in os.environ


def test_process_controller_passes_and_releases_activation_pipe() -> None:
    child_code = (
        "import os; "
        f"fd = int(os.environ[{ACTIVATION_FD_ENV!r}]); "
        "raise SystemExit(0 if os.read(fd, 1) else 2)"
    )
    controller = WorkerProcessController(
        command=[sys.executable, "-c", child_code],
        cwd=PROJECT_ROOT,
        stop_timeout_sec=1.0,
        term_timeout_sec=1.0,
    )

    assert controller.start()
    assert controller.activate()
    assert controller._process.wait(timeout=2.0) == 0
    assert controller.refresh() == 0
    assert not controller.is_running
