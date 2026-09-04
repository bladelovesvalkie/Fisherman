from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WeatherObservation:
    """Provider-neutral weather object returned by an adapter."""

    timestamp: datetime
    latitude: float
    longitude: float
    wind_speed_ms: float | None = None
    wind_gust_ms: float | None = None
    wind_direction_deg: float | None = None
    temperature_c: float | None = None
    pressure_hpa: float | None = None
    humidity_pct: float | None = None
    precipitation_mm: float | None = None


class WeatherProvider:
    def current(self, latitude: float, longitude: float) -> WeatherObservation:
        raise NotImplementedError
