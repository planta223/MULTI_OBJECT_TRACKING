"""NumPy constant-velocity Kalman filter for camera-space translation."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PositionInnovation:
    """A position measurement residual under the current prediction."""

    residual: np.ndarray
    covariance: np.ndarray
    mahalanobis_distance_sq: float


class ConstantVelocityPositionKalmanFilter:
    """Track ``[px, py, pz, vx, vy, vz]`` using real timestamps."""

    _H = np.hstack((np.eye(3), np.zeros((3, 3))))

    def __init__(
        self,
        process_acceleration_std_mps2: float,
        measurement_position_std_m: float,
        initial_position_std_m: float,
        initial_velocity_std_mps: float,
    ) -> None:
        values = {
            "process_acceleration_std_mps2": process_acceleration_std_mps2,
            "measurement_position_std_m": measurement_position_std_m,
            "initial_position_std_m": initial_position_std_m,
            "initial_velocity_std_mps": initial_velocity_std_mps,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        self.process_acceleration_variance = float(
            process_acceleration_std_mps2**2
        )
        self.measurement_covariance = np.eye(3, dtype=np.float64) * (
            measurement_position_std_m**2
        )
        self.initial_position_variance = float(initial_position_std_m**2)
        self.initial_velocity_variance = float(initial_velocity_std_mps**2)
        self._state: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._timestamp_s: Optional[float] = None

    @property
    def initialized(self) -> bool:
        return self._state is not None

    @staticmethod
    def _position(position: np.ndarray) -> np.ndarray:
        values = np.asarray(position, dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("position must be a finite vector with shape (3,).")
        return values

    @staticmethod
    def _timestamp(timestamp_s: float) -> float:
        value = float(timestamp_s)
        if not np.isfinite(value):
            raise ValueError("timestamp_s must be finite.")
        return value

    def _require_initialized(self) -> Tuple[np.ndarray, np.ndarray, float]:
        if self._state is None or self._covariance is None or self._timestamp_s is None:
            raise RuntimeError("Kalman filter must be initialized first.")
        return self._state, self._covariance, self._timestamp_s

    def initialize(self, position: np.ndarray, timestamp_s: float) -> None:
        measured_position = self._position(position)
        timestamp = self._timestamp(timestamp_s)
        self._state = np.concatenate((measured_position, np.zeros(3)))
        self._covariance = np.diag(
            [self.initial_position_variance] * 3
            + [self.initial_velocity_variance] * 3
        ).astype(np.float64)
        self._timestamp_s = timestamp

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:] = np.eye(3) * dt
        return transition

    def _process_covariance(self, dt: float) -> np.ndarray:
        identity = np.eye(3, dtype=np.float64)
        return self.process_acceleration_variance * np.block(
            [
                [(dt**4 / 4.0) * identity, (dt**3 / 2.0) * identity],
                [(dt**3 / 2.0) * identity, (dt**2) * identity],
            ]
        )

    def predict(self, timestamp_s: float) -> np.ndarray:
        state, covariance, previous_timestamp = self._require_initialized()
        timestamp = self._timestamp(timestamp_s)
        dt = timestamp - previous_timestamp
        if dt < 0:
            raise ValueError(
                "Kalman timestamps must be monotonic: "
                f"previous={previous_timestamp}, received={timestamp}."
            )
        if dt > 0:
            transition = self._transition(dt)
            self._state = transition @ state
            self._covariance = (
                transition @ covariance @ transition.T
                + self._process_covariance(dt)
            )
            self._covariance = (
                self._covariance + self._covariance.T
            ) / 2.0
            self._timestamp_s = timestamp
        return self.position

    def innovation(self, measured_position: np.ndarray) -> PositionInnovation:
        state, covariance, _ = self._require_initialized()
        measurement = self._position(measured_position)
        residual = measurement - self._H @ state
        innovation_covariance = (
            self._H @ covariance @ self._H.T + self.measurement_covariance
        )
        try:
            solved = np.linalg.solve(innovation_covariance, residual)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("Innovation covariance is singular.") from error
        mahalanobis_distance_sq = float(residual @ solved)
        return PositionInnovation(
            residual=np.array(residual, copy=True),
            covariance=np.array(innovation_covariance, copy=True),
            mahalanobis_distance_sq=mahalanobis_distance_sq,
        )

    def update(
        self,
        measured_position: np.ndarray,
        timestamp_s: float,
    ) -> PositionInnovation:
        self.predict(timestamp_s)
        innovation = self.innovation(measured_position)
        state, covariance, _ = self._require_initialized()
        gain = covariance @ self._H.T @ np.linalg.inv(innovation.covariance)
        self._state = state + gain @ innovation.residual

        identity = np.eye(6, dtype=np.float64)
        correction = identity - gain @ self._H
        self._covariance = (
            correction @ covariance @ correction.T
            + gain @ self.measurement_covariance @ gain.T
        )
        self._covariance = (self._covariance + self._covariance.T) / 2.0
        return innovation

    @property
    def position(self) -> np.ndarray:
        state, _, _ = self._require_initialized()
        return np.array(state[:3], copy=True)

    @property
    def velocity(self) -> np.ndarray:
        state, _, _ = self._require_initialized()
        return np.array(state[3:], copy=True)

    @property
    def prediction_covariance(self) -> np.ndarray:
        _, covariance, _ = self._require_initialized()
        return np.array(covariance[:3, :3], copy=True)

    @property
    def timestamp_s(self) -> float:
        _, _, timestamp = self._require_initialized()
        return timestamp
