#!/usr/bin/env python3
"""Check the multiclass upsample storage and throughput resource contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ESP_INT8_hls" / "src" / "upsample_unit.cpp"


def lerp_reference(a: int, b: int, weight1: int) -> int:
    return a * (16 - weight1) + b * weight1


def lerp_difference(a: int, b: int, weight1: int) -> int:
    return (a << 4) + (b - a) * weight1


def test_difference_form_is_bit_exact_for_int8_and_all_phases() -> None:
    weights = (0, 1, 3, 5, 7, 9, 11, 13, 15)
    for a in range(-128, 128):
        for b in range(-128, 128):
            for weight1 in weights:
                assert lerp_difference(a, b, weight1) == lerp_reference(a, b, weight1)


def test_hls_keeps_the_low_lut_pipelined_interpolator() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "lerp16_fixed8" not in source
    assert "scale_fixed8_delta" not in source
    assert "top * static_cast<i32_t>(wy0)" in source
    assert "bottom * static_cast<i32_t>(wy1)" in source
    assert "#pragma HLS PIPELINE II=1" in source


def test_previous_logits_use_one_dual_port_lutram() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "s_prev_logits_row[ENCODER_OUT_W]" in source
    assert "BIND_STORAGE variable=s_prev_logits_row type=ram_2p impl=lutram" in source
    assert "s_prev_logits_bank" not in source
    assert "UPSAMPLE_LOGITS_BANKS" not in source


if __name__ == "__main__":
    test_difference_form_is_bit_exact_for_int8_and_all_phases()
    test_hls_keeps_the_low_lut_pipelined_interpolator()
    test_previous_logits_use_one_dual_port_lutram()
    print("upsample LUTRAM resource contract: PASS")
