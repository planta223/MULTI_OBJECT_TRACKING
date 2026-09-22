#!/usr/bin/env python3
"""Preload FoundationPose at unit task 3, activate it at 4, and stop at 6."""

import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Callable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_activation import ACTIVATION_FD_ENV


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-topic",
        default="/recog/unit_task/result",
        help="std_msgs/msg/Int8 unit-task topic.",
    )
    parser.add_argument(
        "--preload-task-id",
        type=int,
        default=3,
        help="Start the worker and preload models at this task.",
    )
    parser.add_argument("--start-task-id", type=int, default=4)
    parser.add_argument("--stop-task-id", type=int, default=6)
    parser.add_argument(
        "--task-timeout-sec",
        type=float,
        default=2.0,
        help="Stop a running worker if task messages disappear for this long.",
    )
    parser.add_argument(
        "--stop-timeout-sec",
        type=float,
        default=8.0,
        help="Seconds to allow graceful SIGINT shutdown before SIGTERM.",
    )
    parser.add_argument(
        "--term-timeout-sec",
        type=float,
        default=2.0,
        help="Seconds to allow SIGTERM shutdown before SIGKILL.",
    )
    parser.add_argument(
        "--worker-cwd",
        type=Path,
        default=PROJECT_ROOT,
        help="Working directory for the worker process.",
    )
    parser.add_argument(
        "worker_command",
        nargs=argparse.REMAINDER,
        help="Worker command following '--'.",
    )
    args = parser.parse_args(argv)

    if args.worker_command[:1] == ["--"]:
        args.worker_command = args.worker_command[1:]
    if not args.worker_command:
        parser.error("provide the worker command after '--'")
    if not args.task_topic:
        parser.error("--task-topic must not be empty")
    for name in ("preload_task_id", "start_task_id", "stop_task_id"):
        value = getattr(args, name)
        if not -128 <= value <= 127:
            parser.error(f"--{name.replace('_', '-')} must fit std_msgs/msg/Int8")
    if not args.preload_task_id < args.start_task_id < args.stop_task_id:
        parser.error(
            "task IDs must satisfy --preload-task-id < --start-task-id "
            "< --stop-task-id"
        )
    for name in ("task_timeout_sec", "stop_timeout_sec", "term_timeout_sec"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.worker_cwd = args.worker_cwd.expanduser().resolve()
    if not args.worker_cwd.is_dir():
        parser.error(f"worker directory does not exist: {args.worker_cwd}")
    return args


class WorkerProcessController:
    """Own one worker process group so CUDA resources die with the worker."""

    def __init__(
        self,
        command: Sequence[str],
        cwd: Path,
        stop_timeout_sec: float,
        term_timeout_sec: float,
        *,
        popen_factory: Callable = subprocess.Popen,
        signal_process_group: Callable[[int, int], None] = os.killpg,
    ) -> None:
        self.command = list(command)
        self.cwd = cwd
        self.stop_timeout_sec = float(stop_timeout_sec)
        self.term_timeout_sec = float(term_timeout_sec)
        self._popen_factory = popen_factory
        self._signal_process_group = signal_process_group
        self._process = None
        self._activation_write_fd: Optional[int] = None

    @property
    def pid(self) -> Optional[int]:
        return None if self._process is None else int(self._process.pid)

    @property
    def is_running(self) -> bool:
        self.refresh()
        return self._process is not None

    def refresh(self) -> Optional[int]:
        if self._process is None:
            return None
        return_code = self._process.poll()
        if return_code is None:
            return None
        self._close_activation_writer()
        self._process = None
        return int(return_code)

    def _close_activation_writer(self) -> None:
        if self._activation_write_fd is None:
            return
        try:
            os.close(self._activation_write_fd)
        except OSError:
            pass
        self._activation_write_fd = None

    def start(self) -> bool:
        if self.is_running:
            return False
        activation_read_fd, activation_write_fd = os.pipe()
        environment = os.environ.copy()
        environment[ACTIVATION_FD_ENV] = str(activation_read_fd)
        try:
            self._process = self._popen_factory(
                self.command,
                cwd=str(self.cwd),
                env=environment,
                start_new_session=True,
                pass_fds=(activation_read_fd,),
            )
        except BaseException:
            os.close(activation_write_fd)
            raise
        finally:
            os.close(activation_read_fd)
        self._activation_write_fd = activation_write_fd
        return True

    def activate(self) -> bool:
        """Release a running worker from its one-shot preload gate."""

        if not self.is_running or self._activation_write_fd is None:
            return False
        write_fd = self._activation_write_fd
        self._activation_write_fd = None
        try:
            os.write(write_fd, b"1")
        except (BrokenPipeError, OSError):
            return False
        finally:
            try:
                os.close(write_fd)
            except OSError:
                pass
        return True

    def _send_signal(self, signum: int) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        try:
            self._signal_process_group(int(self._process.pid), signum)
        except ProcessLookupError:
            pass

    def stop(self) -> bool:
        process = self._process
        if process is None:
            return False
        if process.poll() is None:
            self._send_signal(signal.SIGINT)
            try:
                process.wait(timeout=self.stop_timeout_sec)
            except subprocess.TimeoutExpired:
                self._send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=self.term_timeout_sec)
                except subprocess.TimeoutExpired:
                    self._send_signal(signal.SIGKILL)
                    process.wait()
        self._close_activation_writer()
        self._process = None
        return True


class TaskTriggeredWorker:
    """Translate task transitions into a single worker lifecycle."""

    def __init__(
        self,
        controller: WorkerProcessController,
        preload_task_id: int,
        start_task_id: int,
        stop_task_id: int,
        task_timeout_sec: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = print,
    ) -> None:
        self.controller = controller
        self.preload_task_id = int(preload_task_id)
        self.start_task_id = int(start_task_id)
        self.stop_task_id = int(stop_task_id)
        self.task_timeout_sec = float(task_timeout_sec)
        self._clock = clock
        self._log = log
        self._last_task_id: Optional[int] = None
        self._last_task_time: Optional[float] = None

    def _start_worker(self, reason: str) -> bool:
        try:
            started = self.controller.start()
        except Exception as error:
            self._log(
                f"Worker failed to start during {reason}: "
                f"{type(error).__name__}: {error}"
            )
            return False
        if started:
            self._log(
                f"Worker started for {reason}; pid={self.controller.pid}"
            )
        return started

    def on_task(self, task_id: int) -> None:
        task_id = int(task_id)
        previous = self._last_task_id
        self._last_task_id = task_id
        self._last_task_time = self._clock()
        if task_id != previous:
            self._log(f"Task transition: {previous} -> {task_id}")

        if task_id == self.preload_task_id and previous != self.preload_task_id:
            self._start_worker(f"preload task {task_id}")
            return

        if task_id == self.start_task_id and previous != self.start_task_id:
            if not self.controller.is_running:
                self._log(
                    f"Preload task {self.preload_task_id} was missed; "
                    f"starting worker cold at task {task_id}"
                )
                if not self._start_worker(f"cold activation task {task_id}"):
                    return
            if self.controller.activate():
                self._log(
                    f"Worker activated at task {task_id}; pid={self.controller.pid}"
                )
            else:
                self._log(f"Worker activation failed at task {task_id}")
            return

        if task_id >= self.stop_task_id and self.controller.is_running:
            self._log(f"Stopping worker at task {task_id}")
            self.controller.stop()
            self._log("Worker stopped")

    def tick(self) -> None:
        return_code = self.controller.refresh()
        if return_code is not None:
            self._log(f"Worker exited with status {return_code}")
        if not self.controller.is_running or self._last_task_time is None:
            return
        age = self._clock() - self._last_task_time
        if age > self.task_timeout_sec:
            self._log(
                f"Task topic timed out after {age:.2f}s; stopping worker for safety"
            )
            self.controller.stop()
            self._log("Worker stopped")
            # A resumed task-4 stream is a new activation after a timeout.
            self._last_task_id = None
            self._last_task_time = None

    def shutdown(self) -> None:
        if self.controller.is_running:
            self._log("Supervisor shutdown; stopping worker")
            self.controller.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        import rclpy
        from std_msgs.msg import Int8
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            "The task supervisor requires ROS2 rclpy and std_msgs. "
            "Source /opt/ros/foxy/setup.bash before running it."
        ) from error

    rclpy.init(args=[])
    node = rclpy.create_node("foundationpose_task_supervisor")
    logger = node.get_logger()
    controller = WorkerProcessController(
        command=args.worker_command,
        cwd=args.worker_cwd,
        stop_timeout_sec=args.stop_timeout_sec,
        term_timeout_sec=args.term_timeout_sec,
    )
    supervisor = TaskTriggeredWorker(
        controller=controller,
        preload_task_id=args.preload_task_id,
        start_task_id=args.start_task_id,
        stop_task_id=args.stop_task_id,
        task_timeout_sec=args.task_timeout_sec,
        log=logger.info,
    )
    node.create_subscription(
        Int8,
        args.task_topic,
        lambda message: supervisor.on_task(message.data),
        10,
    )
    node.create_timer(0.2, supervisor.tick)
    logger.info(f"Task topic: {args.task_topic} (std_msgs/msg/Int8)")
    logger.info(
        f"Worker policy: preload={args.preload_task_id}, "
        f"activate={args.start_task_id}, stop={args.stop_task_id}, "
        f"message_timeout={args.task_timeout_sec:g}s"
    )
    logger.info(f"Worker command: {shlex.join(args.worker_command)}")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
