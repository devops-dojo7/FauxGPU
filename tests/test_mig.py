import pytest

from engine.mig import (
    MigRequest,
    list_mig_profiles,
    pack_mig_requests,
    supports_mig,
)


def test_supports_mig_true_only_for_known_gpus():
    for gpu_id in ("a100-80gb-sxm", "a100-40gb-sxm", "h100-sxm", "h100-pcie", "h200-sxm"):
        assert supports_mig(gpu_id) is True
    for gpu_id in ("b200-sxm", "rtx4090", "not-a-real-gpu"):
        assert supports_mig(gpu_id) is False


def test_profile_catalog_matches_the_7_compute_8_memory_split():
    profiles = {p.id: p for p in list_mig_profiles("a100-80gb-sxm")}
    assert profiles["1g.10gb"].compute_slots == 1 and profiles["1g.10gb"].memory_slots == 1
    assert profiles["2g.20gb"].compute_slots == 2 and profiles["2g.20gb"].memory_slots == 2
    assert profiles["3g.40gb"].compute_slots == 3 and profiles["3g.40gb"].memory_slots == 4
    assert profiles["4g.40gb"].compute_slots == 4 and profiles["4g.40gb"].memory_slots == 4
    assert profiles["7g.80gb"].compute_slots == 7 and profiles["7g.80gb"].memory_slots == 8


def test_profile_naming_matches_published_nvidia_conventions():
    # A100 40GB's real published profiles are 1g.5gb/2g.10gb/3g.20gb/4g.20gb/7g.40gb
    ids = {p.id for p in list_mig_profiles("a100-40gb-sxm")}
    assert ids == {"1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"}
    # H200's real published profiles are 1g.18gb/2g.35gb/3g.71gb/4g.71gb/7g.141gb
    ids = {p.id for p in list_mig_profiles("h200-sxm")}
    assert ids == {"1g.18gb", "2g.35gb", "3g.71gb", "4g.71gb", "7g.141gb"}


def test_list_mig_profiles_rejects_unsupported_gpu():
    with pytest.raises(ValueError):
        list_mig_profiles("rtx4090")


def test_single_request_places_on_first_gpu():
    result = pack_mig_requests("a100-80gb-sxm", pool_size=1, requests=[MigRequest("r1", "team-a", "2g.20gb")])
    assert len(result.placements) == 1
    assert result.placements[0].gpu_index == 0
    assert result.unplaced == ()
    assert result.gpus_used == 1


def test_requests_summing_to_full_budget_fill_one_gpu_exactly():
    # 3g.40gb (3 compute/4 memory) + 4g.40gb (4 compute/4 memory) = 7 compute/8 memory = the whole GPU
    requests = [MigRequest("r1", "a", "3g.40gb"), MigRequest("r2", "b", "4g.40gb")]
    result = pack_mig_requests("a100-80gb-sxm", pool_size=1, requests=requests)
    assert len(result.placements) == 2
    assert {p.gpu_index for p in result.placements} == {0}
    assert result.compute_utilization_pct == 100.0
    assert result.memory_utilization_pct == 100.0


def test_overflow_spills_onto_the_next_pool_gpu():
    # Two 7g.80gb requests can't both fit on one GPU (each takes the whole budget)
    requests = [MigRequest("r1", "a", "7g.80gb"), MigRequest("r2", "b", "7g.80gb")]
    result = pack_mig_requests("a100-80gb-sxm", pool_size=2, requests=requests)
    assert len(result.placements) == 2
    assert {p.gpu_index for p in result.placements} == {0, 1}
    assert result.unplaced == ()
    assert result.gpus_used == 2


def test_requests_that_dont_fit_anywhere_are_unplaced_not_dropped_or_raised():
    requests = [MigRequest("r1", "a", "7g.80gb"), MigRequest("r2", "b", "7g.80gb")]
    result = pack_mig_requests("a100-80gb-sxm", pool_size=1, requests=requests)
    assert len(result.placements) == 1
    assert len(result.unplaced) == 1
    assert result.unplaced[0].request_id == "r2"


def test_unsupported_gpu_raises():
    with pytest.raises(ValueError):
        pack_mig_requests("rtx4090", pool_size=1, requests=[])


def test_invalid_profile_for_gpu_raises():
    with pytest.raises(ValueError):
        pack_mig_requests("a100-80gb-sxm", pool_size=1, requests=[MigRequest("r1", "a", "1g.999gb")])


def test_zero_pool_size_rejected():
    with pytest.raises(ValueError):
        pack_mig_requests("a100-80gb-sxm", pool_size=0, requests=[])
