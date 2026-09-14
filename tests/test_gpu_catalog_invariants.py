"""Structural sanity checks on engine/data/gpus.yaml — catches data-entry
mistakes (typos, copy-paste duplicates, an inconsistent price/note pairing)
automatically, rather than relying on manual review every time an entry is
added. Deliberately does not check individual numeric values against vendor
specs — that's a sourcing question for a human, not something a test can
verify.
"""

from engine.gpu_specs import FABRICS, GPUS


def test_no_duplicate_gpu_ids():
    ids = [g.id for g in GPUS.values()]
    assert len(ids) == len(set(ids))


def test_no_duplicate_fabric_ids():
    ids = [f.id for f in FABRICS.values()]
    assert len(ids) == len(set(ids))


def test_every_gpu_has_vendor_and_name():
    for g in GPUS.values():
        assert g.vendor.strip(), f"{g.id} has an empty vendor"
        assert g.name.strip(), f"{g.id} has an empty name"


def test_core_numeric_fields_are_positive():
    for g in GPUS.values():
        assert g.vram_gb > 0, g.id
        assert g.mem_bandwidth_gbps > 0, g.id
        assert g.bf16_tflops > 0, g.id
        assert g.tdp_watts > 0, g.id
        assert g.price_per_hr_usd >= 0, g.id


def test_idle_watts_never_exceeds_tdp():
    for g in GPUS.values():
        assert 0 <= g.idle_watts <= g.tdp_watts, g.id


def test_fp8_never_slower_than_bf16_when_both_published():
    # This table's own convention (see gpus.yaml header comment): fp8 is
    # either a distinct, faster precision or left null — never slower.
    for g in GPUS.values():
        if g.fp8_tflops is not None:
            assert g.fp8_tflops >= g.bf16_tflops, g.id


def test_free_gpus_always_explain_why():
    # price_per_hr_usd == 0 means "not a $/hr rental" (captive silicon, an
    # unreleased roadmap part, or a purchase-only system) — the table's own
    # convention is to always say which via price_note, never leave it
    # unexplained.
    for g in GPUS.values():
        if g.price_per_hr_usd == 0:
            assert g.price_note, f"{g.id} is priced at $0/hr with no price_note explaining why"


def test_nvlink_bandwidth_positive_when_present():
    for g in GPUS.values():
        if g.nvlink_gbps is not None:
            assert g.nvlink_gbps > 0, g.id
