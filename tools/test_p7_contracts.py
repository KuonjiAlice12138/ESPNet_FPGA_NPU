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
ARTIFACT_DIR = ROOT / "hw_artifacts" / "binary2_int8_h256w512_r2_v4"

sys.path.insert(0, str(TOOL_DIR))

import eval_val_hw_masks_fullres as eval_fullres
import export_int8_hw_blob as blob_tools
from export_exec_prefix_params import build_exec_prefix_blobs
from hw_int8_math import (
    hls_activation_quant_dequant,
    hls_add_bypass_dequant,
    hls_conv2d_i8_nchw,
    hls_round_away_from_zero,
    hls_round_shift_int,
)
from hw_param_replay import (
    ParamBlob,
    decode_window_loader_runs as replay_decode_window_loader_runs,
)


def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise AssertionError(f"actual={actual.cpu().tolist()} expected={expected.cpu().tolist()}")


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

    # Historical 0809 artifacts used a 120 KiB binary2 reservation, while
    # current H256/W512 export reserves the larger city20-safe capacity.  The
    # contract is that both the recorded legacy reservation and the current
    # exporter capacity can contain the packed payload, not that old metadata
    # must equal today's constant.
    assert audit["packed_weights"]["wbuf_bytes"] >= audit["packed_weights"]["total_bytes"]
    assert blob_tools.WBUF_BYTES >= audit["packed_weights"]["total_bytes"]
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


def test_exec_prefix_param_contract() -> None:
    source = (ARTIFACT_DIR / "PARAM.BIN").read_bytes()
    prefixes, records = build_exec_prefix_blobs(source)
    blob = ParamBlob(ARTIFACT_DIR / "PARAM.BIN")
    exec_offset = blob.header["exec_plan_offset"]
    active_count = next(i for i, entry in enumerate(blob.exec_plan) if entry.kind == blob_tools.EXEC_END)

    assert active_count == 15
    assert len(prefixes) == active_count + 1
    assert len(records) == active_count + 1

    with tempfile.TemporaryDirectory() as td:
        for prefix_count, data in enumerate(prefixes):
            assert len(data) == len(source)
            prefix_path = Path(td) / f"P{prefix_count:02d}.BIN"
            prefix_path.write_bytes(data)
            parsed = ParamBlob(prefix_path)
            first_end = next(
                i for i, entry in enumerate(parsed.exec_plan)
                if entry.kind == blob_tools.EXEC_END
            )
            assert first_end == prefix_count
            assert records[prefix_count]["active_exec_count"] == prefix_count
            assert records[prefix_count]["file"] == prefix_path.name

            changed = [i for i, (lhs, rhs) in enumerate(zip(source, data)) if lhs != rhs]
            if prefix_count < active_count:
                assert changed == [exec_offset + prefix_count * 4]
                assert data[changed[0]] == blob_tools.EXEC_END
            else:
                assert changed == []
                assert data == source


def test_exec_prefix_app_contract() -> None:
    config = (ROOT / "ESP_INT8_app" / "src" / "app_config.h").read_text(encoding="utf-8")
    main_src = (ROOT / "ESP_INT8_app" / "src" / "main.c").read_text(encoding="utf-8")
    app_yaml = (ROOT / "ESP_INT8_app" / "src" / "app.yaml").read_text(encoding="utf-8")
    hal_header = (
        ROOT / "ESP_INT8_app" / "src" / "hal" / "int8_npu.h"
    ).read_text(encoding="utf-8")

    assert '#define INT8_APP_ENABLE_EXEC_PREFIX_PROFILE 0U' in config
    assert '#define INT8_APP_ENABLE_DUAL_SINGLE_TEST 1U' in config
    assert '#define INT8_APP_EXEC_PREFIX_COUNT 16U' in config
    assert (
        '#define INT8_APP_EXEC_PREFIX_REFERENCE_FULL_RTL_CYCLES 0ULL'
        in config
    )
    assert (
        '#define INT8_APP_EXEC_PREFIX_TARGET_MAX_RTL_CYCLES 0ULL'
        in config
    )
    assert 'INT8-BOARD-20260923-P7-H256W512-R4R-DUAL-PROF' in config
    assert "platform_r4_0922" in app_yaml
    assert "run_exec_prefix_profile" in main_src
    assert "EXEC_PREFIX_CYCLES_BEGIN" in main_src
    assert "PREFIX_CUM" in main_src
    assert "PREFIX_DELTA" in main_src
    assert "APP: prefix baseline" in main_src
    assert "APP: prefix target" in main_src
    assert "stage_counter_read_status" in hal_header
    assert "stage_counter_read_current" in hal_header


def _window_request_columns(desc, issue: int) -> list[int]:
    paired = bool(desc.flags & blob_tools.WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)
    pixel0 = issue * (2 if paired else 1)
    columns = [
        pixel0 * desc.stride - desc.padding + kw * desc.dilation
        for kw in range(3)
    ]
    if paired and pixel0 + 1 < desc.out_w:
        columns.extend(
            (pixel0 + 1) * desc.stride - desc.padding + kw * desc.dilation
            for kw in range(3)
        )
    return columns


def _expected_update_deltas(desc, issue: int, update_mask: int) -> list[int]:
    columns = _window_request_columns(desc, issue)
    base = issue * 2 * desc.stride - desc.padding
    return sorted(
        {
            column - base
            for request, column in enumerate(columns)
            if update_mask & (1 << request)
        }
    )


def _expand_loader_runs(runs) -> list[int]:
    return [
        start_delta + offset
        for start_delta, count in runs
        for offset in range(count)
    ]


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
            assert desc.loader_class == blob_tools.WIN_LOADER_3X3_NARROW
            assert desc.loader_request_cols == 3
            assert desc.loader_warmup_issues == dilation
            assert desc.loader_warmup_new_cols == 3
            assert desc.loader_steady_new_cols == 1
            assert desc.loader_words_per_col == 3
            assert desc.loader_warmup_mask == 0x07
            assert desc.loader_steady_mask == 0x04
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
        assert desc.loader_class == blob_tools.WIN_LOADER_3X3_WIDE
        assert desc.loader_request_cols == 3
        assert desc.loader_warmup_issues == 1
        assert desc.loader_warmup_new_cols == 3
        assert desc.loader_steady_new_cols == 1
        assert desc.loader_words_per_col == 3 * desc.cache_chunks
        assert desc.loader_warmup_mask == 0x07
        assert desc.loader_steady_mask == 0x04

    uops = blob_tools.build_uops()
    schedules, _commands, schedule_ids, _audit = blob_tools.build_window_schedule_sections(uops)
    reuse_contracts = {
        (
            sched.mode,
            sched.stride,
            sched.dilation,
            sched.in_c,
            sched.out_w,
        ): (
            sched.reserved[blob_tools.WINDOW_LOADER_ROW_REUSE_WORD]
            & blob_tools.WINDOW_ROW_REUSE_MODE_MASK,
            (
                sched.reserved[blob_tools.WINDOW_LOADER_ROW_REUSE_WORD]
                >> blob_tools.WINDOW_ROW_REUSE_WORDS_SHIFT
            )
            & blob_tools.WINDOW_ROW_REUSE_WORDS_MASK,
        )
        for sched in schedules
        if (
            sched.reserved[blob_tools.WINDOW_LOADER_ROW_REUSE_WORD]
            & blob_tools.WINDOW_ROW_REUSE_MODE_MASK
        )
    }
    assert reuse_contracts == {
        (
            blob_tools.WIN_MODE_3X3_STAGED_C3,
            2,
            1,
            3,
            256,
        ): (blob_tools.WINDOW_ROW_REUSE_STRIDE2_KEEP1, 48),
    }
    assert all(
        (
            sched.reserved[blob_tools.WINDOW_LOADER_ROW_REUSE_WORD]
            >> blob_tools.WINDOW_ROW_REUSE_RESERVED_SHIFT
        )
        == 0
        for sched in schedules
    )
    for uop in uops:
        if uop.opcode == blob_tools.UOP_CONV and uop.kernel == 3:
            sched = schedules[schedule_ids[uop.param_id]]
            assert sched.mode == blob_tools.window_mode_for(uop.in_c, 3)
            assert sched.stride == uop.stride
            assert sched.dilation == uop.dilation
            assert sched.padding == uop.padding
            assert sched.out_w == blob_tools.conv_out_dim(uop.in_w, uop.stride)
            assert sched.loader_class in (
                blob_tools.WIN_LOADER_3X3_NARROW,
                blob_tools.WIN_LOADER_3X3_WIDE,
            )
            assert sched.loader_request_cols in (3, 6)
            assert sched.loader_warmup_new_cols >= sched.loader_steady_new_cols
            assert sched.loader_words_per_col == 3 * sched.cache_chunks

    first_layer = schedules[schedule_ids[0]]
    assert first_layer.loader_request_cols == 6
    assert first_layer.loader_warmup_issues == 1
    assert first_layer.loader_warmup_new_cols == 5
    assert first_layer.loader_steady_new_cols == 4
    assert first_layer.loader_warmup_mask == 0x37
    assert first_layer.loader_steady_mask == 0x36
    assert blob_tools.decode_window_loader_runs(first_layer.reserved, warmup=True) == ((0, 5),)
    assert blob_tools.decode_window_loader_runs(first_layer.reserved, warmup=False) == ((1, 4),)
    assert blob_tools.decode_window_loader_phase_split(
        first_layer.reserved, warmup=True
    ) == 3
    assert blob_tools.decode_window_loader_phase_split(
        first_layer.reserved, warmup=False
    ) == 2
    assert replay_decode_window_loader_runs(first_layer.reserved, warmup=True) == ((0, 5),)
    assert replay_decode_window_loader_runs(first_layer.reserved, warmup=False) == ((1, 4),)

    packed_first_layer = blob_tools.pack_window_sched_desc(first_layer)
    assert len(packed_first_layer) == 128
    assert tuple(packed_first_layer[100:108]) == (
        first_layer.loader_class,
        first_layer.loader_request_cols,
        first_layer.loader_warmup_issues,
        first_layer.loader_warmup_new_cols,
        first_layer.loader_steady_new_cols,
        first_layer.loader_words_per_col,
        first_layer.loader_warmup_mask,
        first_layer.loader_steady_mask,
    )
    assert tuple(
        int.from_bytes(packed_first_layer[108 + index * 2 : 110 + index * 2], "little")
        for index in range(10)
    ) == tuple(first_layer.reserved)

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
        narrow_paired_3x3 = (
            uop.kernel == 3
            and paired
            and sched.loader_class == blob_tools.WIN_LOADER_3X3_NARROW
        )
        if narrow_paired_3x3:
            warmup_runs = blob_tools.decode_window_loader_runs(
                sched.reserved, warmup=True
            )
            steady_runs = blob_tools.decode_window_loader_runs(
                sched.reserved, warmup=False
            )
            assert 1 <= len(warmup_runs) <= 3
            assert 1 <= len(steady_runs) <= 3
            assert _expand_loader_runs(warmup_runs) == _expected_update_deltas(
                sched, 0, sched.loader_warmup_mask
            )
            steady_issue = max(sched.loader_warmup_issues, 1)
            assert _expand_loader_runs(steady_runs) == _expected_update_deltas(
                sched, steady_issue, sched.loader_steady_mask
            )
            warmup_split = blob_tools.decode_window_loader_phase_split(
                sched.reserved, warmup=True
            )
            steady_split = blob_tools.decode_window_loader_phase_split(
                sched.reserved, warmup=False
            )
            assert warmup_split <= sched.loader_warmup_new_cols
            assert steady_split <= sched.loader_steady_new_cols
            if sched.stride == 2 and sched.dilation == 1:
                assert warmup_split > 0
                assert steady_split > 0
            else:
                assert warmup_split == 0
                assert steady_split == 0
        else:
            assert all(int(value) == 0 for value in sched.reserved)
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


def test_serial_row_schedule_contracts() -> None:
    uops = blob_tools.build_uops()
    schedules, _commands, schedule_ids, _audit = blob_tools.build_window_schedule_sections(uops)
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
    conv_exec, row_consumers, _fixed, plan, _block5, coverage = blob_tools.build_exec_plan_sections(
        uops, schedule_ids, weight_offsets, weight_report
    )
    assert plan
    for desc in conv_exec:
        sched = schedules[desc.window_sched_id]
        known_flags = (
            blob_tools.WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2
            | blob_tools.WINDOW_SCHED_FLAG_ODD_TAIL
        )
        assert (sched.flags & ~known_flags) == 0
        narrow_paired_3x3 = (
            sched.kernel == 3
            and sched.loader_class == blob_tools.WIN_LOADER_3X3_NARROW
            and bool(sched.flags & blob_tools.WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)
        )
        if narrow_paired_3x3:
            assert blob_tools.decode_window_loader_runs(sched.reserved, warmup=True)
            assert blob_tools.decode_window_loader_runs(sched.reserved, warmup=False)
        else:
            assert all(int(value) == 0 for value in sched.reserved)
        assert int(desc.reserved) == 0
        assert row_consumers[desc.row_consumer_id]

    schedule_audit = coverage["conv_segment_schedule"]
    assert len(schedule_audit) == len(conv_exec)
    assert all(record["segment_out_w"] == 0 for record in schedule_audit)
    assert all(record["segments_per_row"] == 0 for record in schedule_audit)
    assert all(record["cache_restarts_per_row"] == 0 for record in schedule_audit)
    assert coverage["compiled_cache_restarts"] == 0
    assert coverage["compiled_segment_calls"] == 0


def test_serial_row_wingen_source_contract() -> None:
    conv_source = (ROOT / "ESP_INT8_hls" / "src" / "conv_engine.cpp").read_text(encoding="utf-8")
    win_source = (ROOT / "ESP_INT8_hls" / "src" / "win_gen.cpp").read_text(encoding="utf-8")

    assert "scheduled_window_generator_segment" not in conv_source
    assert "scheduled_window_generator_segment" not in win_source
    assert win_source.count("void scheduled_window_generator_row(") == 1
    assert conv_source.count("scheduled_window_generator_row(") == 2

    # A row-local issue_count field is valid in the narrowed fallback config;
    # reject only the retired segment API and its derived loop variables.
    for legacy_token in (
        "u16_t issue_begin",
        "bool row_begin",
        "issue_begin_i",
        "issue_count_i",
        "local_issue_i",
        "local_ow_i",
        "if (row_begin)",
    ):
        assert legacy_token not in win_source

    # Preserve the P7 wide-channel cache, but let PARAM drive all replacement
    # decisions so no runtime column-tag inference remains in the hot path.
    assert "s_wide_cache" in win_source
    assert "s_wide_col_tag" not in win_source
    assert "s_narrow_col_tag" not in win_source

    # Round 4 keeps one FMBUF reader and one assembler-owned rolling spatial
    # cache in the bounded DATAFLOW region. Full-window transfer remains only
    # for schedule classes that did not opt into the incremental protocol.
    for required in (
        "window_row_loader",
        "window_row_assembler",
        "scheduled_3x3_window_row_pipeline",
        "window_column_meta_t",
        "window_load_word_t",
        "narrow_paired_issue_plan_t",
        "window_row_loader_narrow_paired",
        "window_row_loader_narrow_reuse",
        "window_loader_emit_narrow_cached_window",
        "window_loader_emit_narrow_reuse_window",
        "window_row_assembler_read_narrow_window",
        "select_narrow_transfer_protocol",
        "window_row_loader_narrow_incremental_c12",
        "window_row_loader_narrow_incremental_c3_reuse",
        "window_loader_emit_narrow_direct_run<4>",
        "window_loader_emit_narrow_direct_run<2>",
        "window_loader_emit_narrow_reuse_run<5>",
        "window_loader_emit_narrow_reuse_run<4>",
        "window_assembler_consume_c12_incremental",
        "window_assembler_consume_c3_reuse_incremental",
        "narrow_incremental_cache_t incremental_cache",
        "s_narrow_row_bank0",
        "s_narrow_row_bank1",
        "s_narrow_row_bank2",
    ):
        assert required in win_source
    assert "static narrow_incremental_cache_t" not in win_source
    assert "narrow_issue_stream" not in win_source
    assert "ESP_INT8_CSIM_VALIDATE_INCREMENTAL_TRANSFER" in win_source
    assert "stage_narrow_3x3_window_pair" not in win_source
    for obsolete in (
        "prepare_narrow_issue_windows",
        "update_wide_direct_cache",
        "load_narrow_cache_column",
        "load_wide_cache_column",
        "stage_narrow_3x3_window_select",
        "s_narrow_cache_row",
        "window_loader_emit_narrow_run",
        "window_loader_emit_narrow_run_row",
    ):
        assert obsolete not in win_source
    assert "#pragma HLS STREAM variable=column_meta_stream depth=16" in win_source
    assert "#pragma HLS STREAM variable=load_word_stream depth=16" in win_source
    assert "#pragma HLS STREAM variable=act_stream0 depth=40" in conv_source
    assert "#pragma HLS BIND_STORAGE variable=act_stream0 type=fifo impl=lutram" in conv_source


def test_round4r_wingen_lane_mask_contract() -> None:
    win_source = (ROOT / "ESP_INT8_hls" / "src" / "win_gen.cpp").read_text(
        encoding="utf-8"
    )

    # Round 4's 256-bit dynamic shift/subtract became a 16-CARRY8 setup path.
    # Build the lane-valid mask once in the row configuration stage and gate
    # each output byte independently in the loader instead.
    assert "win_low_byte_mask" not in win_source
    assert "static_cast<act_vec_t>(1) <<" not in win_source
    assert "u8_t narrow_valid_c;" in win_source
    assert "u8_t last_chunk_valid_c;" in win_source
    assert "u32_t narrow_lane_mask;" in win_source
    assert "u32_t last_chunk_lane_mask;" in win_source
    assert "win_build_lane_mask" in win_source
    assert "win_apply_lane_mask" in win_source
    mask_helper = win_source.split(
        "static act_vec_t win_apply_lane_mask", 1
    )[1].split("struct window_row_cfg_t", 1)[0]
    assert "if (lane_mask[lane])" not in mask_helper
    assert "masked.range" not in mask_helper
    assert "for (int bit = 0; bit < 8; ++bit)" in mask_helper
    assert "masked[bit_index] = word[bit_index] & lane_mask[lane];" in mask_helper
    assert "cfg.in_c" not in win_source


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
                block5_row_group = fixed.flags & blob_tools.FIXED_FLAG_BLOCK5_ROW_GROUP
                if block5_row_group:
                    assert not (fixed.flags & blob_tools.FIXED_FLAG_CBLOCK_MAJOR), (entry, fixed)
                    assert not (fixed.flags & blob_tools.FIXED_FLAG_ROW_CONTIGUOUS_STORE), (entry, fixed)
                    continue
                if row_contig:
                    assert not (fixed.flags & blob_tools.FIXED_FLAG_CBLOCK_MAJOR), (
                        entry,
                        fixed,
                        "row-contiguous fixed output must remain pixel-major",
                    )
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
                    assert fixed.flags & blob_tools.FIXED_FLAG_CBLOCK_MAJOR, (
                        entry,
                        fixed,
                        "aligned fixed output must carry an explicit compiled cblock-major loop mode",
                    )
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


def test_avgpool_single_pixel_schedule_contract() -> None:
    source = (ROOT / "ESP_INT8_hls" / "src" / "avgpool_unit.cpp").read_text(
        encoding="utf-8"
    )

    for required in (
        "quantize_c3_avg_pixel(",
        "sum_c3_cached_pixel(",
        "store_c3_pixel_to_group_words(",
        "for (int pix = 0; pix < AVGPOOL_C3_GROUP_PIXELS; ++pix)",
    ):
        assert required in source

    for obsolete in (
        "quantize_c3_avg_pixel_pair",
        "sum_c3_cached_pixel_pair",
        "avgpool_group_buffer_t",
        "avgpool_append_c3_pixel",
    ):
        assert obsolete not in source


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
        ("exec_prefix_param_contract", test_exec_prefix_param_contract),
        ("exec_prefix_app_contract", test_exec_prefix_app_contract),
        ("window_schedule_contracts", test_window_schedule_contracts),
        ("serial_row_schedule_contracts", test_serial_row_schedule_contracts),
        ("serial_row_wingen_source_contract", test_serial_row_wingen_source_contract),
        ("round4r_wingen_lane_mask_contract", test_round4r_wingen_lane_mask_contract),
        ("store_layout_schedule_contracts", test_store_layout_schedule_contracts),
        ("weight_pack_contract", test_weight_pack_contract),
        ("avgpool_single_pixel_schedule_contract", test_avgpool_single_pixel_schedule_contract),
        ("eval_contracts", test_eval_contracts),
    ]
    for name, fn in tests:
        run(name, fn)
    print("P7 Python contract tests passed")


if __name__ == "__main__":
    main()
