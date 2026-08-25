import pytest
from fastapi import HTTPException

from api.routers.mig import get_mig_profiles, list_mig_gpus, pack
from api.schemas import MigPackRequest, MigRequestIn


def test_list_mig_gpus_returns_known_gpus_with_friendly_names():
    gpus = list_mig_gpus()
    ids = {g.gpu_id for g in gpus}
    assert "a100-80gb-sxm" in ids
    assert "h200-sxm" in ids
    by_id = {g.gpu_id: g.gpu_name for g in gpus}
    assert by_id["h100-sxm"] == "H100 SXM5 80GB"


def test_get_mig_profiles_returns_known_profiles():
    profiles = get_mig_profiles("a100-80gb-sxm")
    ids = {p.id for p in profiles}
    assert "7g.80gb" in ids


def test_get_mig_profiles_404_for_non_mig_gpu():
    with pytest.raises(HTTPException) as exc_info:
        get_mig_profiles("rtx4090")
    assert exc_info.value.status_code == 404


def test_pack_returns_placements_for_valid_request():
    req = MigPackRequest(
        gpu_id="a100-80gb-sxm",
        pool_size=1,
        requests=[MigRequestIn(request_id="r1", tenant="team-a", profile_id="2g.20gb")],
    )
    res = pack(req)
    assert len(res.placements) == 1
    assert res.placements[0].tenant == "team-a"
    assert res.unplaced == []


def test_pack_400_on_invalid_profile():
    req = MigPackRequest(
        gpu_id="a100-80gb-sxm",
        pool_size=1,
        requests=[MigRequestIn(request_id="r1", tenant="team-a", profile_id="9g.999gb")],
    )
    with pytest.raises(HTTPException) as exc_info:
        pack(req)
    assert exc_info.value.status_code == 400


def test_pack_400_on_unsupported_gpu():
    req = MigPackRequest(gpu_id="rtx4090", pool_size=1, requests=[])
    with pytest.raises(HTTPException) as exc_info:
        pack(req)
    assert exc_info.value.status_code == 400
