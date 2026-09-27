"""Carbon/energy accounting: converts a training run's already-computed
energy consumption (engine.power's watt figures, accumulated over
engine.cost's total_time_hours — see api/routers/calculate.py's
total_energy_kwh) into an estimated CO2e footprint.

Grid carbon intensity varies enormously by region and by the minute (grid
mix shifts with renewable output) — there is no single "correct" number.
This module uses illustrative annual-average intensities per region
(broadly consistent with published national/grid-operator averages, not
live data), the same "illustrative snapshot, refresh periodically" caveat
this project's own engine/data/gpus.yaml price_per_hr_usd column already
carries. Pure Python, no I/O, no framework dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

# gCO2e per kWh, illustrative annual-average grid carbon intensity by region.
# Real intensity varies hour-to-hour with the live generation mix (e.g. a
# solar-heavy grid at noon vs. the same grid at night) — these are coarse,
# single-number stand-ins for "roughly how clean is this region's grid",
# not a live-data feed. Sourced from the rough public range for each grid's
# annual average (fossil-heavy grids high, hydro/nuclear-heavy grids low);
# treat as a teaching approximation and refresh periodically, same as the
# GPU price table.
GRID_CARBON_INTENSITY_G_PER_KWH: dict[str, float] = {
    "us-avg": 386.0,  # US national grid average
    "us-west-hydro": 100.0,  # Pacific Northwest-style hydro-heavy grid
    "us-texas": 400.0,  # ERCOT, gas-heavy with growing wind/solar
    "eu-avg": 275.0,  # EU-27 average
    "eu-france-nuclear": 60.0,  # France's nuclear-heavy grid
    "eu-germany": 350.0,  # Coal/gas-heavier despite renewables buildout
    "uk": 200.0,
    "china": 550.0,  # coal-heavy national average
    "india": 700.0,  # coal-heavy national average
    "asia-pacific-avg": 500.0,
    "global-avg": 475.0,  # world average generation mix
}

DEFAULT_REGION = "global-avg"


@dataclass(frozen=True)
class CarbonEstimate:
    region: str
    grid_intensity_g_per_kwh: float
    energy_kwh: float
    co2e_kg: float
    # Illustrative equivalences, the same rough public conversion factors
    # commonly used to make a kg-CO2e figure legible (a car's tailpipe
    # emissions per km, one round-trip transatlantic flight seat) — not
    # precise, just enough to make the number feel like something.
    equivalent_car_km: float
    equivalent_flights_ny_london: float


# ~192 gCO2e/km for an average passenger car (well-to-wheel, illustrative).
_CAR_G_PER_KM = 192.0
# ~986 kg CO2e for one economy seat, New York <-> London round trip (illustrative).
_FLIGHT_NY_LONDON_KG = 986.0


def grid_intensity(region: str = DEFAULT_REGION) -> float:
    try:
        return GRID_CARBON_INTENSITY_G_PER_KWH[region]
    except KeyError:
        raise ValueError(
            f"Unknown region: {region!r}. Known: {sorted(GRID_CARBON_INTENSITY_G_PER_KWH)}"
        ) from None


def estimate_carbon(energy_kwh: float, region: str = DEFAULT_REGION) -> CarbonEstimate:
    intensity = grid_intensity(region)
    co2e_kg = energy_kwh * intensity / 1000.0
    return CarbonEstimate(
        region=region,
        grid_intensity_g_per_kwh=intensity,
        energy_kwh=energy_kwh,
        co2e_kg=co2e_kg,
        equivalent_car_km=(co2e_kg * 1000.0) / _CAR_G_PER_KM,
        equivalent_flights_ny_london=co2e_kg / _FLIGHT_NY_LONDON_KG,
    )
