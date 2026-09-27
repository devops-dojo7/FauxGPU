from engine.carbon import DEFAULT_REGION, estimate_carbon, grid_intensity


def test_known_region_intensities_resolve():
    assert grid_intensity("us-avg") > 0
    assert grid_intensity("eu-france-nuclear") > 0
    assert grid_intensity(DEFAULT_REGION) > 0


def test_unknown_region_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown region"):
        grid_intensity("mars")


def test_co2e_scales_linearly_with_energy():
    low = estimate_carbon(energy_kwh=10.0, region="us-avg")
    high = estimate_carbon(energy_kwh=20.0, region="us-avg")
    assert high.co2e_kg == 2 * low.co2e_kg


def test_zero_energy_yields_zero_carbon():
    result = estimate_carbon(energy_kwh=0.0)
    assert result.co2e_kg == 0.0
    assert result.equivalent_car_km == 0.0
    assert result.equivalent_flights_ny_london == 0.0


def test_cleaner_grid_yields_less_carbon_for_same_energy():
    dirty = estimate_carbon(energy_kwh=100.0, region="india")
    clean = estimate_carbon(energy_kwh=100.0, region="eu-france-nuclear")
    assert clean.co2e_kg < dirty.co2e_kg


def test_equivalences_are_positive_and_consistent():
    result = estimate_carbon(energy_kwh=500.0, region="global-avg")
    assert result.equivalent_car_km > 0
    assert result.equivalent_flights_ny_london > 0
    # More energy -> proportionally more of both equivalences.
    double = estimate_carbon(energy_kwh=1000.0, region="global-avg")
    assert double.equivalent_car_km == 2 * result.equivalent_car_km
    assert double.equivalent_flights_ny_london == 2 * result.equivalent_flights_ny_london
