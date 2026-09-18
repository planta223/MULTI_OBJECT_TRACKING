"""CPU-only tests for task-triggered FoundationPose worker lifecycle."""

from pathlib import Path

import pytest

from scripts.run_task_supervisor import TaskTriggeredWorker, parse_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeController:
    def __init__(self) -> None:
        self.running = False
        self.start_calls = 0
        self.stop_calls = 0
        self.pid = None

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self) -> bool:
        if self.running:
            return False
        self.running = True
        self.start_calls += 1
        self.pid = 1234
        return True

    def stop(self) -> bool:
        if not self.running:
            return False
        self.running = False
        self.stop_calls += 1
        self.pid = None
        return True

    def refresh(self):
        return None


def make_supervisor(controller, now):
    return TaskTriggeredWorker(
        controller=controller,
        start_task_id=4,
        stop_task_id=6,
        task_timeout_sec=2.0,
        clock=lambda: now[0],
        log=lambda message: None,
    )


def test_worker_starts_once_at_four_runs_through_five_and_stops_at_six() -> None:
    controller = FakeController()
    now = [0.0]
    supervisor = make_supervisor(controller, now)

    for task_id in (0, 1, 2, 3):
        supervisor.on_task(task_id)
    assert controller.start_calls == 0

    supervisor.on_task(4)
    supervisor.on_task(4)
    supervisor.on_task(5)
    assert controller.start_calls == 1
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


def test_later_task_stops_worker_if_task_six_message_was_missed() -> None:
    controller = FakeController()
    supervisor = make_supervisor(controller, [0.0])

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
    assert args.worker_cwd == PROJECT_ROOT
    assert args.worker_command[-2:] == ["--camera-type", "ros_zed"]
