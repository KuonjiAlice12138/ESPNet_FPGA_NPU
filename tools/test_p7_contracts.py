#!/usr/bin/env python3
"""Aggregated lightweight P7 Python-side contract tests.

This replaces the older one-test-per-file scripts and keeps the active
PARAM v4 / hardware-QAT checks in a single entry point.
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
ARTIFACT_DIR = ROOT / "hw_artifacts" / "sched_v4_p7_0702"
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
    hard_checks = audit["hard_checks"]
    memory_lifetime = audit["memory_lifetime"]

    assert blob.header["magic"] == 0x544E4945
    assert blob.header["version"] == 4
    assert blob.header["tensor_desc_count"] == expected_sizes["tensor_desc"] // 16
    assert blob.header["conv_exec_desc_count"] == expected_sizes["conv_exec_desc"] // 36
    assert blob.header["exec_plan_count"] == expected_sizes["exec_plan"] // 4
    assert blob.header["window_sched_count"] == expected_sizes["window_sched_desc"] // 128
    assert blob.header["window_cmd_count"] == audit["window_pack_cmd_count"]
    assert blob.header["window_cmd_count"] == 0
    assert blob.header["row_consumer_count"] == expected_sizes["row_consumer_desc"] // 16
    assert blob.header["block5_sched_offset"] == expected_offsets["block5_sched_desc"]

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

    assert audit["packed_weights"]["wbuf_bytes"] == 120 * 1024
    assert hard_checks["packed_weight_fits_wbuf"] is True
    assert hard_checks["pool2_alias_lifetime_safe"] is True
    assert hard_checks["pool2_alias_safe"] is True
    alias = memory_lifetime["aliases"][0]
    assert alias["name"] == "POOL2_OVER_POOL1_FMBUF"
    assert alias["virtual_bank"] == "BANK_BRAM_SCR1"
    assert alias["target_bank"] == "BANK_FMEM0"
    assert alias["base_offset"] == blob_tools.FMBUF_POOL2_ALIAS_BASE
    pool1_lifetime = memory_lifetime["tensor_lifetimes"]["1"]
    pool2_lifetime = memory_lifetime["tensor_lifetimes"]["8"]
    assert pool1_lifetime["last_read"] < pool2_lifetime["first_write"]


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
    got = replay_prefix(blob, ARTIFACT_DIR / "input_q.bin", stop_logical_uop=2).read_tensor(3)
    expected = load_golden_nhwc(QAT_DIR / "golden_sample" / "b1_bn" / "output_int.npy")
    result = compare_i8_arrays(got, expected)
    assert result["mismatches"] == 0, result


def test_prefix_export_smoke() -> None:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        report = export_prefix(
            artifact_dir=ARTIFACT_DIR,
            qat_dir=QAT_DIR,
            out_dir=out_dir,
            stop_logical_uop=2,
            tensor_ids=[3],
        )
        assert report["stop_logical_uop"] == 2
        assert (out_dir / "T03_replay_nhwc.npy").exists()
        assert (out_dir / "T03_replay_nhwc.bin").exists()
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["tensors"]["3"]["shape"] == [256, 512, 19]
        assert "qat_golden_compare" in manifest["tensors"]["3"]


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
    for dilation in (1, 2, 4, 8, 16):
        for in_c in (3, 12, 19, 25):
            desc, commands = blob_tools.make_window_pack_schedule(
                in_c=in_c,
                kernel=3,
                stride=1,
                dilation=dilation,
                padding=dilation,
                out_w=128,
            )
            assert desc.mode == blob_tools.window_mode_for(in_c, 3)
            assert desc.cache_chunks == 1
            assert desc.cache_col_slots == 4 * dilation
            assert desc.k_tiles == (in_c * 9 + blob_tools.TK - 1) // blob_tools.TK
            assert len(commands) < desc.k_tiles * blob_tools.TK
            for kt in range(desc.k_tiles):
                begin = desc.kt_cmd_base[kt]
                end = desc.kt_cmd_base[kt + 1]
                assert end - begin <= blob_tools.MAX_PACK_CMDS_PER_KT
                assert sum(cmd.byte_count for cmd in commands[begin:end]) == min(
                    blob_tools.TK, in_c * 9 - kt * blob_tools.TK
                )

    for in_c in (28, 64, 128, 131):
        desc, _commands = blob_tools.make_window_pack_schedule(
            in_c=in_c, kernel=3, stride=1, dilation=1, padding=1, out_w=128
        )
        assert desc.cache_chunks == (in_c + blob_tools.TK - 1) // blob_tools.TK
        assert desc.cache_col_slots == 4

    uops = blob_tools.build_uops()
    schedules, _commands, schedule_ids, _audit = blob_tools.build_window_schedule_sections(uops)
    for uop in uops:
        if uop.opcode == blob_tools.UOP_CONV and uop.kernel == 3:
            sched = schedules[schedule_ids[uop.param_id]]
            assert sched.mode == blob_tools.window_mode_for(uop.in_c, 3)
            assert sched.stride == uop.stride
            assert sched.dilation == uop.dilation
            assert sched.padding == uop.padding
            assert sched.out_w == blob_tools.conv_out_dim(uop.in_w, uop.stride)

    paired_params = {
        uop.param_id
        for uop in uops
        if uop.opcode == blob_tools.UOP_CONV
        and uop.out_c <= 16
        and uop.out_c > 0
        and blob_tools.window_mode_supports_pixel_parallel(
            blob_tools.window_mode_for(
                uop.in_c,
                uop.kernel,
                blob_tools.tensor_supports_aligned_1x1_read(uop.src0),
            )
        )
    }
    assert paired_params
    for uop in uops:
        if uop.opcode != blob_tools.UOP_CONV:
            continue
        sched = schedules[schedule_ids[uop.param_id]]
        paired = bool(sched.flags & blob_tools.WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)
        assert paired == (uop.param_id in paired_params), (uop.param_id, uop.out_c, sched.flags)
        odd_tail = bool(sched.flags & blob_tools.WINDOW_SCHED_FLAG_ODD_TAIL)
        assert odd_tail == (paired and (sched.out_w & 1) != 0)
    # P7F keeps L20/L2B0 materialized as compact C64 tensors. U21 therefore
    # should use the fast aligned 1x1 path instead of the old wide-slice packed
    # reader.
    assert schedules[schedule_ids[7]].mode == blob_tools.WIN_MODE_1X1_ALIGNED
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

    conv_exec, row_consumers, fixed_exec, exec_plan, block5_sched, _coverage = blob_tools.build_exec_plan_sections(
        uops, schedule_ids, weight_offsets, weight_report
    )
    assert fixed_exec
    assert block5_sched
    for sched in block5_sched:
        assert sched.branch_count == 5
        assert sched.pattern in (blob_tools.BLOCK5_PATTERN_L2_C16_4C12, blob_tools.BLOCK5_PATTERN_L3_C28_4C25)
        assert sched.first_branch_conv_id + 4 < len(conv_exec)
        assert sched.row_group_h == 64
    for entry in exec_plan:
        if entry.kind in (
            blob_tools.EXEC_POOL,
            blob_tools.EXEC_BLOCK_AFFINE,
            blob_tools.EXEC_BLOCK_ADD_AFFINE,
        ):
            assert entry.desc_id < len(fixed_exec)
            assert fixed_exec[entry.desc_id].kind == entry.kind
            if entry.kind in (blob_tools.EXEC_BLOCK_AFFINE, blob_tools.EXEC_BLOCK_ADD_AFFINE):
                fixed = fixed_exec[entry.desc_id]
                dst = blob_tools.TENSOR_DESC_BY_ID[fixed.dst_tensor]
                row_contig = fixed.flags & blob_tools.FIXED_FLAG_ROW_CONTIGUOUS_STORE
                if row_contig:
                    assert fixed.valid_c == dst.c == dst.phys_c, (
                        entry,
                        fixed,
                        dst,
                        "row-contiguous fixed output must materialize a full compact row",
                    )
                    assert dst.c_offset == 0, (entry, fixed, dst, "row-contiguous fixed output cannot be a slice")
                    assert dst.base_offset % blob_tools.TM == 0, (
                        entry,
                        fixed,
                        dst,
                        "row-contiguous fixed output base must be 32B aligned",
                    )
                    assert (dst.w * fixed.valid_c) % blob_tools.TM == 0, (
                        entry,
                        fixed,
                        dst,
                        "row-contiguous fixed output row bytes must be 32B aligned",
                    )
                else:
                    assert fixed.valid_c % blob_tools.TM == 0, (
                        entry,
                        fixed,
                        "fixed output channel count is not full-tile",
                    )
                    assert dst.phys_c % blob_tools.TM == 0, (
                        entry,
                        fixed,
                        dst,
                        "fixed output row stride is not 32B aligned",
                    )
                    assert (dst.base_offset + dst.c_offset) % blob_tools.TM == 0, (
                        entry,
                        fixed,
                        dst,
                        "fixed output first tile is not 32B aligned",
                    )
                    assert fixed.valid_c <= dst.c, (entry, fixed, dst, "fixed output exceeds tensor logical channels")

    executed_conv_desc_ids = {
        int(entry.desc_id)
        for entry in exec_plan
        if entry.kind == blob_tools.EXEC_CONV
    }
    for desc_id, desc in enumerate(conv_exec):
        consumer = row_consumers[desc.row_consumer_id]
        if desc_id not in executed_conv_desc_ids and consumer.mode == blob_tools.ROW_CONSUMER_NONE:
            assert consumer.reserved0 == blob_tools.STORE_LAYOUT_NONE, (desc.param_id, consumer)
            continue
        if consumer.mode == blob_tools.ROW_CONSUMER_UPSAMPLE_OUT:
            assert consumer.reserved0 == blob_tools.STORE_LAYOUT_NONE, (desc.param_id, consumer)
            continue
        assert consumer.mode in (
            blob_tools.ROW_CONSUMER_NONE,
            blob_tools.ROW_CONSUMER_CAT_AFFINE_STORE,
        )
        assert consumer.reserved0 in (
            blob_tools.STORE_LAYOUT_NONE,
            blob_tools.STORE_LAYOUT_COMPACT_C12,
            blob_tools.STORE_LAYOUT_COMPACT_C19,
            blob_tools.STORE_LAYOUT_COMPACT_C25,
        ), (desc.param_id, consumer)
        if desc_id in executed_conv_desc_ids:
            assert consumer.reserved0 != blob_tools.STORE_LAYOUT_NONE, (desc.param_id, consumer)


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
