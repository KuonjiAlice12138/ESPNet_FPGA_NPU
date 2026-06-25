#!/usr/bin/env python3
"""Aggregated lightweight P7 Python-side contract tests.

This replaces the older one-test-per-file scripts and keeps the active
PARAM v3 / hardware-QAT checks in a single entry point.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
ARTIFACT_DIR = ROOT / "hw_artifacts" / "sched_v3_single_p7_hwconv_0623"
QAT_DIR = ROOT / "quantized_artifacts_hw_constrained_qat_p7_hwconv_0623"

sys.path.insert(0, str(TOOL_DIR))

import eval_val_hw_masks_fullres as eval_fullres
import export_int8_hw_blob as blob_tools
from audit_p7_precision_contract import detect_round_mode, inspect_qat_hook_source
from export_p7_param_replay_prefix import export_prefix
from hw_int8_math import (
    hls_activation_quant_dequant,
    hls_add_bypass_dequant,
    hls_conv2d_i8_nchw,
    hls_round_away_from_zero,
    hls_round_shift_int,
)
from hw_param_replay import ParamBlob, _store_slice, compare_i8_arrays, replay_prefix


def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise AssertionError(f"actual={actual.cpu().tolist()} expected={expected.cpu().tolist()}")


def load_golden_nhwc(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"{path} shape={arr.shape}; expected NCHW batch=1")
    return np.ascontiguousarray(np.transpose(arr[0], (1, 2, 0))).astype(np.int8)


def test_hw_int8_math() -> None:
    values = torch.tensor([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=torch.int64)
    expected = torch.tensor([-3, -2, -2, -1, -1, 0, 1, 1, 2, 2, 3], dtype=torch.int64)
    assert_tensor_equal(hls_round_shift_int(values, 1), expected)

    half_values = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=torch.float32)
    half_expected = torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.float32)
    assert_tensor_equal(hls_round_away_from_zero(half_values), half_expected)

    act = hls_activation_quant_dequant(torch.tensor([[-3.0, -0.2, 0.2, 40.0]]), scale=0.25, relu=True)
    assert_tensor_equal(act, torch.tensor([[0.0, 0.0, 0.25, 31.75]], dtype=torch.float32))

    add = hls_add_bypass_dequant(
        torch.tensor([12.0, -12.0, 2.5], dtype=torch.float32),
        torch.tensor([8.0, -8.0, -4.0], dtype=torch.float32),
        scale=0.125,
        relu=False,
    )
    assert_tensor_equal(add, torch.tensor([15.875, -16.0, -1.5], dtype=torch.float32))


def test_param_parser_contract() -> None:
    audit = json.loads((ARTIFACT_DIR / "param_audit.json").read_text(encoding="utf-8"))
    blob = ParamBlob(ARTIFACT_DIR / "PARAM.BIN")

    expected_offsets = audit["section_offsets"]
    expected_sizes = audit["section_sizes"]

    assert blob.header["magic"] == 0x544E4945
    assert blob.header["version"] == 3
    assert blob.header["tensor_desc_count"] == expected_sizes["tensor_desc"] // 16
    assert blob.header["conv_exec_desc_count"] == expected_sizes["conv_exec_desc"] // 36
    assert blob.header["exec_plan_count"] == expected_sizes["exec_plan"] // 4
    assert blob.header["window_sched_count"] == expected_sizes["window_sched_desc"] // 96
    assert blob.header["window_cmd_count"] == audit["window_pack_cmd_count"]
    assert blob.header["row_consumer_count"] == expected_sizes["row_consumer_desc"] // 16

    for key, expected in expected_offsets.items():
        header_key = {
            "window_pack_cmd": "window_cmd_offset",
            "row_consumer_desc": "row_consumer_offset",
        }.get(key, f"{key}_offset")
        if header_key in blob.header:
            assert blob.header[header_key] == expected, (header_key, blob.header[header_key], expected)

    assert len(blob.tensor_desc) == blob.header["tensor_desc_count"]
    assert len(blob.conv_exec) == blob.header["conv_exec_desc_count"]
    assert len(blob.window_sched) == blob.header["window_sched_count"]
    assert len(blob.pack_cmds) == blob.header["window_cmd_count"]
    assert len(blob.row_consumer) == blob.header["row_consumer_count"]
    assert len(blob.exec_plan) == blob.header["exec_plan_count"]


def test_prefix_replay_contract() -> None:
    blob = ParamBlob(ARTIFACT_DIR / "PARAM.BIN")

    first = _store_slice(None, (128, 256, 12), 0, np.zeros((128, 256, 12), dtype=np.int8))
    second = _store_slice(first, (64, 128, 25), 0, np.ones((64, 128, 25), dtype=np.int8))
    assert second.shape == (64, 128, 25)
    assert int(second.sum()) == 64 * 128 * 25

    replay = replay_prefix(blob, ARTIFACT_DIR / "input_q.bin", stop_logical_uop=20)
    got = replay.read_tensor(5)
    expected = load_golden_nhwc(QAT_DIR / "golden_sample" / "level2_0_bn" / "output_int.npy")
    result = compare_i8_arrays(got, expected)
    assert result["mismatches"] == 0, result


def test_conv_forward_contract() -> None:
    blob = ParamBlob(ARTIFACT_DIR / "PARAM.BIN")
    input_nhwc = np.fromfile(ARTIFACT_DIR / "input_q.bin", dtype=np.int8).reshape(512, 1024, 3)
    input_nchw = torch.from_numpy(np.transpose(input_nhwc, (2, 0, 1))[None, ...].copy())

    weight = torch.from_numpy(np.load(QAT_DIR / "layers" / "level1_conv" / "weight_int8.npy").astype(np.int8))
    bias, mult, shift = blob.conv_qparam[0]
    got = hls_conv2d_i8_nchw(
        input_i8=input_nchw,
        weight_i8=weight,
        bias_i32=torch.tensor(bias[:16], dtype=torch.int64),
        mult_i32=torch.tensor(mult[:16], dtype=torch.int64),
        shift_u8=torch.tensor(shift[:16], dtype=torch.int64),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        relu=True,
    )

    expected_nhwc = replay_prefix(blob, ARTIFACT_DIR / "input_q.bin", stop_logical_uop=3).read_tensor(2)[:, :, :16]
    expected = np.ascontiguousarray(np.transpose(expected_nhwc, (2, 0, 1))[None, ...]).astype(np.int8)
    result = compare_i8_arrays(got.numpy().astype(np.int8), expected)
    assert result["mismatches"] == 0, result


def test_prefix_export_smoke() -> None:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        report = export_prefix(
            artifact_dir=ARTIFACT_DIR,
            qat_dir=QAT_DIR,
            out_dir=out_dir,
            stop_logical_uop=4,
            tensor_ids=[2, 3],
        )
        assert report["stop_logical_uop"] == 4
        assert (out_dir / "T02_replay_nhwc.npy").exists()
        assert (out_dir / "T02_replay_nhwc.bin").exists()
        assert (out_dir / "T03_replay_nhwc.npy").exists()
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["tensors"]["2"]["shape"] == [256, 512, 19]
        assert manifest["tensors"]["3"]["shape"] == [256, 512, 19]
        assert "qat_golden_compare" in manifest["tensors"]["2"]


def test_precision_audit_contract() -> None:
    round_mode = detect_round_mode(ROOT / "ESP_INT8_hls" / "include" / "npu_q.hpp")
    assert round_mode == "round_away_negative", round_mode

    hook_report = inspect_qat_hook_source(TOOL_DIR / "hw_int8_math.py")
    assert hook_report["hls_conv2d_i8_helper"] is True
    assert hook_report["hls_requant_i32_helper"] is True
    assert hook_report["activation_hooks"] is True
    assert hook_report["floatfunctional_add_patch"] is True
    assert hook_report["floatfunctional_cat_patch"] is True
    assert hook_report["conv2d_forward_patch"] is True


def test_window_schedule_contracts() -> None:
    for in_c in (12, 19, 25):
        desc, commands = blob_tools.make_window_pack_schedule(in_c=in_c, kernel=3)
        assert desc.mode == blob_tools.WIN_MODE_SMALLC_3X3_STAGED
        assert desc.k_tiles == (in_c * 9 + blob_tools.TK - 1) // blob_tools.TK
        assert len(commands) < desc.k_tiles * blob_tools.TK
        for kt in range(desc.k_tiles):
            begin = desc.kt_cmd_base[kt]
            end = desc.kt_cmd_base[kt + 1]
            assert end - begin <= blob_tools.MAX_PACK_CMDS_PER_KT
            assert sum(cmd.byte_count for cmd in commands[begin:end]) == min(
                blob_tools.TK, in_c * 9 - kt * blob_tools.TK
            )

    uops = blob_tools.build_uops()
    schedules, _commands, schedule_ids, _audit = blob_tools.build_window_schedule_sections(uops)
    # U21 reads T_L20_ACT as a slice inside a 131-channel physical row. It is
    # not 32-byte aligned for most pixels, so PARAM must not select aligned 1x1.
    assert schedules[schedule_ids[7]].mode == blob_tools.WIN_MODE_1X1_PACKED
    # Later 1x1 stages read fully aligned physical layouts and should keep the
    # faster aligned mode.
    assert schedules[schedule_ids[19]].mode == blob_tools.WIN_MODE_1X1_ALIGNED
    assert schedules[schedule_ids[25]].mode == blob_tools.WIN_MODE_1X1_ALIGNED

    for uop in uops:
        if uop.opcode != blob_tools.UOP_CONV or uop.kernel != 1:
            continue
        mode = schedules[schedule_ids[uop.param_id]].mode
        source_aligned = blob_tools.tensor_supports_aligned_1x1_read(uop.src0)
        if mode == blob_tools.WIN_MODE_1X1_ALIGNED:
            assert source_aligned, (uop.param_id, uop.src0)
            assert uop.in_c % blob_tools.TK == 0, (uop.param_id, uop.in_c)
        else:
            assert mode == blob_tools.WIN_MODE_1X1_PACKED
            assert (uop.in_c % blob_tools.TK != 0) or not source_aligned


def test_store_layout_schedule_contracts() -> None:
    uops = blob_tools.build_uops()
    _schedules, _commands, schedule_ids, _audit = blob_tools.build_window_schedule_sections(uops)
    conv_uops = [uop for uop in uops if uop.opcode == blob_tools.UOP_CONV]
    weight_offsets = {uop.param_id: uop.param_id * 1024 for uop in conv_uops}
    weight_report = [
        {
            "param_id": uop.param_id,
            "packed_words": ((uop.in_c * uop.kernel * uop.kernel + blob_tools.TK - 1) // blob_tools.TK)
            * uop.out_c,
            "k_tiles": (uop.in_c * uop.kernel * uop.kernel + blob_tools.TK - 1) // blob_tools.TK,
        }
        for uop in conv_uops
    ]

    conv_exec, row_consumers, _exec_plan, _coverage = blob_tools.build_exec_plan_sections(
        uops, schedule_ids, weight_offsets, weight_report
    )

    for desc in conv_exec:
        consumer = row_consumers[desc.row_consumer_id]
        if consumer.mode == blob_tools.ROW_CONSUMER_UPSAMPLE_OUT:
            assert consumer.reserved0 == blob_tools.STORE_LAYOUT_NONE, (desc.param_id, consumer)
            continue
        assert consumer.mode in (
            blob_tools.ROW_CONSUMER_NONE,
            blob_tools.ROW_CONSUMER_STORE,
            blob_tools.ROW_CONSUMER_ADD_STORE,
        )
        assert consumer.reserved0 != blob_tools.STORE_LAYOUT_NONE, (desc.param_id, consumer)
        assert consumer.reserved0 in blob_tools.STORE_LAYOUT_NAMES, (desc.param_id, consumer)


def test_weight_pack_contract() -> None:
    weight = np.arange(2 * 3 * 3 * 3, dtype=np.int16).reshape(2, 3, 3, 3).astype(np.int8)
    packed = blob_tools.pack_single_conv_weights(weight, out_c=2, in_c=3, kernel=3)
    assert len(packed.words) == 2
    expected = []
    for k in range(27):
        kh, kw, cin = blob_tools.linear_k_to_spatial_c(k, 3, 3)[1:]
        expected.append(int(weight[0, cin, kh, kw]) & 0xFF)
    expected.extend([0] * (blob_tools.TK - 27))
    assert packed.words[0] == bytes(expected)


def test_eval_contracts() -> None:
    try:
        eval_fullres.validate_eval_contract("M%04d.BIN", allow_reference_mask_eval=False)
    except ValueError:
        pass
    else:
        raise AssertionError("reference-mask pattern should be rejected by default")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "O0000.BIN"
        path.write_bytes(b"\x00" * (64 * 128))
        result = eval_fullres.validate_output_set(Path(td), "O%04d.BIN", 0, 1)
        assert result["missing"] == []
        assert result["present"] == []
        assert len(result["wrong_size"]) == 1


def run(name: str, fn) -> None:
    fn()
    print(f"[PASS] {name}")


def main() -> None:
    tests = [
        ("hw_int8_math", test_hw_int8_math),
        ("param_parser_contract", test_param_parser_contract),
        ("prefix_replay_contract", test_prefix_replay_contract),
        ("conv_forward_contract", test_conv_forward_contract),
        ("prefix_export_smoke", test_prefix_export_smoke),
        ("precision_audit_contract", test_precision_audit_contract),
        ("window_schedule_contracts", test_window_schedule_contracts),
        ("store_layout_schedule_contracts", test_store_layout_schedule_contracts),
        ("weight_pack_contract", test_weight_pack_contract),
        ("eval_contracts", test_eval_contracts),
    ]
    for name, fn in tests:
        run(name, fn)
    print("P7 Python contract tests passed")


if __name__ == "__main__":
    main()
