from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    latitude: float
    longitude: float
    pressure_hpa: float | None = None
    temperature_c: float | None = None
    humidity_pct: float | None = None
    wind_speed_ms: float | None = None
    wind_gust_ms: float | None = None
    wind_direction_deg: float | None = None
    gps_speed_ms: float | None = None
    heading_deg: float | None = None
    acceleration_x: float | None = None
    acceleration_y: float | None = None
    acceleration_z: float | None = None
    gyro_x: float | None = None
    gyro_y: float | None = None
    gyro_z: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items()}


REQUIRED_COLUMNS = {"timestamp", "latitude", "longitude"}


def validate_columns(columns: list[str] | set[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
