#!/usr/bin/env python3
"""Offline replay of P7 PARAM.BIN using HLS-equivalent integer math.

Reads PARAM v4 blob, packed weights, window schedule, exec plan,
and replays the full encoder forward pass using only INT8/INT32
fixed-point arithmetic aligned with ``include/npu_q.hpp``.

Usage:
    python tools/hw_param_replay.py \\
        --param D:/ESP_INT8/hw_artifacts/sched_v4_p7_0702/PARAM.BIN \\
        --input D:/ESP_INT8/hw_artifacts/sched_v4_p7_0702/input_q.bin \\
        --dump-dir D:/ESP_INT8/ESP_INT8_hls/hls_work_p7_hwqat_fullres_dump_u39_0623/hls/csim/build \\
        --layer U40
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_contract import DeploymentGeometry
from hw_int8_math import (
    clamp_i8,
    hls_round_shift_int,
    INT8_MIN,
    INT8_MAX,
)

# ── constants ──────────────────────────────────────────────
PARAM_BLOB_MAGIC = 0x544E4945
TK = 32
TM = 32
SECTION_ALIGN = 64
FMBUF_BYTES = 0x1F0000
FMBUF_URAM_BYTES = 0x1C0000
FMBUF_POOL1_BASE = 0x1C0000
FMBUF_POOL_TMP_BASE = 0x1D8000
BANK_BRAM_SCR0 = 0x80
BANK_BRAM_SCR1 = 0x81
MAX_BLOCK5_SCHED_COUNT = 8
WINDOW_LOADER_RUN_MAX = 3
WINDOW_LOADER_RESERVED_WORDS = 10


def decode_window_loader_runs(
    reserved: Sequence[int], *, warmup: bool
) -> Tuple[Tuple[int, int], ...]:
    words = tuple(int(value) for value in reserved)
    if len(words) != WINDOW_LOADER_RESERVED_WORDS:
        raise ValueError(
            f"window loader run descriptor has {len(words)} words, "
            f"expected {WINDOW_LOADER_RESERVED_WORDS}"
        )
    count_word = 0 if warmup else 4
    first_run_word = 1 if warmup else 5
    run_count = words[count_word]
    if run_count > WINDOW_LOADER_RUN_MAX:
        raise ValueError(f"window loader run count exceeds {WINDOW_LOADER_RUN_MAX}: {run_count}")
    runs: List[Tuple[int, int]] = []
    for index in range(run_count):
        word = words[first_run_word + index]
        start_delta = word & 0xFF
        if start_delta & 0x80:
            start_delta -= 0x100
        count = (word >> 8) & 0xFF
        if count <= 0:
            raise ValueError(f"window loader run has zero count: 0x{word:04X}")
        runs.append((start_delta, count))
    if any(words[first_run_word + run_count : first_run_word + WINDOW_LOADER_RUN_MAX]):
        raise ValueError("window loader unused run words must be zero")
    return tuple(runs)


def decode_window_loader_phase_split(
    reserved: Sequence[int], *, warmup: bool
) -> int:
    words = tuple(int(value) for value in reserved)
    if len(words) != WINDOW_LOADER_RESERVED_WORDS:
        raise ValueError(
            f"window loader run descriptor has {len(words)} words, "
            f"expected {WINDOW_LOADER_RESERVED_WORDS}"
        )
    word = words[8]
    return (word & 0xFF) if warmup else ((word >> 8) & 0xFF)

# ── enums ──────────────────────────────────────────────────
UOP_NOP, UOP_LOAD_FM, UOP_CONV, UOP_POOL = 0, 1, 2, 3
UOP_ADD, UOP_AFFINE, UOP_STORE, UOP_END = 4, 5, 6, 15

EXEC_NOP, EXEC_CONV, EXEC_POOL = 0, 1, 2
EXEC_BLOCK_AFFINE = 6
EXEC_BLOCK_ADD_AFFINE = 7
EXEC_END = 255
REMOVED_P6_EXEC_KINDS = {3, 4, 5}

FIXED_FLAG_BLOCK5_AFFINE = 1 << 0
FIXED_FLAG_BLOCK5_ADD_AFFINE = 1 << 1
FIXED_FLAG_BLOCK5_ROW_GROUP = 1 << 2
FIXED_FLAG_CBLOCK_MAJOR = 1 << 3
FIXED_FLAG_ROW_CONTIGUOUS_STORE = 1 << 7

BLOCK5_ADD_TENSOR_NONE = 0xFF
BLOCK5_PATTERN_INVALID = 0
BLOCK5_PATTERN_L2_C16_4C12 = 1
BLOCK5_PATTERN_L3_C28_4C25 = 2
BLOCK5_FINALIZER_INVALID = 0
BLOCK5_FINALIZER_L2 = 1
BLOCK5_FINALIZER_L3 = 2

ROW_CONSUMER_NONE, ROW_CONSUMER_STORE = 0, 1
ROW_CONSUMER_ADD_STORE, ROW_CONSUMER_ADD_AFFINE_STORE = 2, 3
ROW_CONSUMER_UPSAMPLE_OUT, ROW_CONSUMER_AFFINE_STORE = 4, 5
ROW_CONSUMER_CAT_AFFINE_STORE = 6
ROW_CONSUMER_ADD_STORE_BLOCK_ADD_AFFINE = 7

WIN_MODE_3X3_RESERVED = 1
WIN_MODE_1X1_ALIGNED = 2
WIN_MODE_1X1_PACKED = 3
WIN_MODE_3X3_STAGED_C3 = 4
WIN_MODE_3X3_STAGED_C12 = 5
WIN_MODE_3X3_STAGED_C19 = 6
WIN_MODE_3X3_STAGED_C25 = 7
WIN_MODE_3X3_STAGED_C131 = 8
WIN_MODE_3X3_STAGED_C28 = 9
WIN_MODE_3X3_STAGED_C64 = 10
WIN_MODE_3X3_STAGED_C128 = 11

WIN_LOADER_3X3_NARROW = 1
WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2 = 1 << 0
WINDOW_LOADER_ROW_REUSE_WORD = 9
WINDOW_ROW_REUSE_NONE = 0
WINDOW_ROW_REUSE_STRIDE1_KEEP2 = 1
WINDOW_ROW_REUSE_STRIDE2_KEEP1 = 2
WINDOW_ROW_REUSE_MODE_MASK = 0x3
WINDOW_ROW_REUSE_WORDS_SHIFT = 2
WINDOW_ROW_REUSE_WORDS_MASK = 0x7F
WINDOW_ROW_REUSE_RESERVED_SHIFT = 9
WINGEN_NARROW_ROW_WORDS = 96


def window_row_reuse_contract_valid(
    desc: "WindowSchedDesc",
    conv: "ConvExecDesc | None" = None,
    src: "TensorDesc | None" = None,
) -> bool:
    if len(desc.reserved) != WINDOW_LOADER_RESERVED_WORDS:
        return False
    word = desc.reserved[WINDOW_LOADER_ROW_REUSE_WORD]
    if word >> WINDOW_ROW_REUSE_RESERVED_SHIFT:
        return False
    mode = desc.row_reuse_mode
    words = desc.packed_words_per_source_row
    if mode == WINDOW_ROW_REUSE_NONE:
        return words == 0
    common = (
        mode == WINDOW_ROW_REUSE_STRIDE2_KEEP1
        and desc.mode == WIN_MODE_3X3_STAGED_C3
        and desc.kernel == 3
        and desc.stride == 2
        and desc.in_c == 3
        and desc.dilation == 1
        and desc.loader_class == WIN_LOADER_3X3_NARROW
        and bool(desc.flags & WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)
        and 0 < words <= WINGEN_NARROW_ROW_WORDS
    )
    if not common or conv is None or src is None:
        return False
    row_bytes = src.w * src.phys_c
    out_w = (conv.in_w + 2 * conv.padding - 3) // 2 + 1
    return (
        src.h > 0 and src.w > 0 and src.c == 3 and src.phys_c == 3
        and src.elem_bytes == 1 and src.c_offset == 0 and src.base_offset % 32 == 0
        and src.h == conv.in_h and src.w == conv.in_w
        and row_bytes % 32 == 0 and row_bytes // 32 == words
        and conv.out_c > 0 and conv.out_c <= 16
        and all(getattr(conv, field) == getattr(desc, field)
                for field in ("in_c", "kernel", "stride", "dilation", "padding", "k_tiles"))
        and desc.k_tiles == 1 and desc.out_w == out_w
        and bool(desc.flags & 2) == bool(out_w % 2)
    )
STAGED_3X3_MODES = {
    WIN_MODE_3X3_STAGED_C3,
    WIN_MODE_3X3_STAGED_C12,
    WIN_MODE_3X3_STAGED_C19,
    WIN_MODE_3X3_STAGED_C25,
    WIN_MODE_3X3_STAGED_C28,
    WIN_MODE_3X3_STAGED_C64,
    WIN_MODE_3X3_STAGED_C128,
    WIN_MODE_3X3_STAGED_C131,
}

ACT_NONE, ACT_RELU = 0, 1
FLAG_RELU_EN = 1 << 1


def unpack_block5_reserved1(reserved1: int) -> Tuple[int, int, int, int]:
    raw = int(reserved1) & 0xFFFFFFFF
    return raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF, (raw >> 24) & 0xFF


def block5_source_channels(pattern: int) -> Tuple[int, int, int, int, int]:
    if pattern == BLOCK5_PATTERN_L2_C16_4C12:
        return (16, 12, 12, 12, 12)
    if pattern == BLOCK5_PATTERN_L3_C28_4C25:
        return (28, 25, 25, 25, 25)
    raise ValueError(f"unsupported BLOCK5 source pattern: {pattern}")

# ── dataclasses ────────────────────────────────────────────
@dataclass
class TensorDesc:
    bank_id: int = 0
    base_offset: int = 0
    h: int = 0
    w: int = 0
    c: int = 0
    reserved0: int = 0
    reserved1: int = 0
    elem_bytes: int = 1
    @property
    def phys_c(self) -> int:
        return self.reserved0 if self.reserved0 else self.c
    @property
    def c_offset(self) -> int:
        return self.reserved1
    def elem_offset(self, h: int, w: int, c: int) -> int:
        return (h * self.w + w) * self.phys_c + self.c_offset + c


@dataclass
class WindowPackCmd:
    spatial_id: int = 0
    src_c_begin: int = 0
    dst_lane_begin: int = 0
    byte_count: int = 0
    flags: int = 1


@dataclass
class WindowSchedDesc:
    mode: int = 0
    kernel: int = 1
    stride: int = 1
    dilation: int = 1
    padding: int = 0
    cache_chunks: int = 0
    cache_col_slots: int = 0
    flags: int = 0
    in_c: int = 0
    out_w: int = 0
    k_tiles: int = 0
    cmd_base: int = 0
    cmd_count: int = 0
    kt_cmd_base: List[int] = None
    loader_class: int = 0
    loader_request_cols: int = 0
    loader_warmup_issues: int = 0
    loader_warmup_new_cols: int = 0
    loader_steady_new_cols: int = 0
    loader_words_per_col: int = 0
    loader_warmup_mask: int = 0
    loader_steady_mask: int = 0
    reserved: List[int] = None
    def __post_init__(self):
        if self.kt_cmd_base is None:
            self.kt_cmd_base = []
        if self.reserved is None:
            self.reserved = []

    @property
    def row_reuse_mode(self) -> int:
        if len(self.reserved) <= WINDOW_LOADER_ROW_REUSE_WORD:
            return WINDOW_ROW_REUSE_NONE
        return (
            self.reserved[WINDOW_LOADER_ROW_REUSE_WORD]
            & WINDOW_ROW_REUSE_MODE_MASK
        )

    @property
    def packed_words_per_source_row(self) -> int:
        if len(self.reserved) <= WINDOW_LOADER_ROW_REUSE_WORD:
            return 0
        return (
            self.reserved[WINDOW_LOADER_ROW_REUSE_WORD]
            >> WINDOW_ROW_REUSE_WORDS_SHIFT
        ) & WINDOW_ROW_REUSE_WORDS_MASK


@dataclass
class ConvExecDesc:
    param_id: int = 0
    qparam_id: int = 0
    window_sched_id: int = 0
    row_consumer_id: int = 0
    in_h: int = 0
    in_w: int = 0
    in_c: int = 0
    out_c: int = 0
    kernel: int = 1
    stride: int = 1
    dilation: int = 1
    padding: int = 0
    src_tensor: int = 0
    dst_tensor: int = 0
    dst_c_offset: int = 0
    valid_c: int = 0
    packed_weight_word_offset: int = 0
    k_tiles: int = 0
    weight_words: int = 0
    flags: int = 0
    reserved: int = 0


@dataclass
class RowConsumerDesc:
    mode: int = 0
    add_other_tensor: int = 255
    store_dst_tensor: int = 255
    add_qparam_id: int = 0
    store_c_offset: int = 0
    valid_c: int = 0
    alias_tensor: int = 255
    affine_param_id: int = 0
    affine_block_base: int = 0
    act_type: int = 0
    store_layout: int = 0
    affine_c_offset: int = 0


@dataclass
class ExecPlanEntry:
    kind: int = 0
    desc_id: int = 0
    logical_uop_id: int = 0
    flags: int = 0


@dataclass
class FixedExecDesc:
    kind: int = 0
    src0_tensor: int = 255
    src1_tensor: int = 255
    dst_tensor: int = 255
    param_id: int = 0
    add_param_id: int = 0
    act_type: int = 0
    flags: int = 0
    in_h: int = 0
    in_w: int = 0
    in_c: int = 0
    out_c: int = 0
    kernel: int = 0
    stride: int = 0
    dilation: int = 0
    padding: int = 0
    c_offset: int = 0
    valid_c: int = 0
    qparam_id: int = 0
    reserved0: int = 0
    reserved1: int = 0


@dataclass
class Block5SchedDesc:
    pattern: int = 0
    branch_count: int = 0
    first_branch_conv_id: int = 0
    first_window_sched_id: int = 0
    first_conv_qparam_id: int = 0
    src_tensor: int = 255
    dst_tensor: int = 255
    add_tensor: int = 255
    chain_add_qparam_id0: int = 0
    chain_add_qparam_id1: int = 0
    chain_add_qparam_id2: int = 0
    residual_add_qparam_id: int = 0
    affine_param_id: int = 0
    affine_block_count: int = 0
    finalizer_kind: int = 0
    scratch_region: int = 0
    out_h: int = 0
    out_w: int = 0
    row_group_h: int = 0
    valid_c: int = 0


@dataclass
class AddQParam:
    mult: int = 1
    shift: int = 0
    act_type: int = 0
    requant_bypass: int = 1
    src_scale_id_a: int = 0
    src_scale_id_b: int = 0
    dst_scale_id: int = 0


@dataclass
class PoolQParam:
    kernel: int = 2
    stride: int = 2
    same_scale: int = 1
    act_type: int = 0
    src_scale_id: int = 0
    dst_scale_id: int = 0
    mult: int = 1
    shift: int = 0


# ── blob parser ────────────────────────────────────────────
class ParamBlob:
    def __init__(self, path: Path):
        self.data = path.read_bytes()
        self._parse()

    def _u8(self, off): return self.data[off]
    def _u16(self, off): return struct.unpack_from("<H", self.data, off)[0]
    def _u32(self, off): return struct.unpack_from("<I", self.data, off)[0]
    def _i32(self, off): return struct.unpack_from("<i", self.data, off)[0]

    def _parse(self):
        words = struct.unpack_from("<32I", self.data, 0)
        self.header = {
            "magic": words[0],
            "version": words[1],
            "tensor_desc_count": words[2],
            "scale_desc_count": words[3],
            "conv_exec_desc_count": words[4],
            "affine_desc_count": words[5],
            "add_desc_count": words[6],
            "pool_desc_count": words[7],
            "uop_count": words[8],
            "exec_plan_count": words[9],
            "tensor_desc_offset": words[10],
            "scale_desc_offset": words[11],
            "uop_offset": words[16],
            "weight_packed_offset": words[17],
            "conv_qparam_offset": words[18],
            "affine_qparam_offset": words[19],
            "add_qparam_offset": words[20],
            "pool_qparam_offset": words[21],
            "conv_exec_desc_offset": words[22],
            "window_sched_offset": words[23],
            "window_cmd_offset": words[24],
            "row_consumer_offset": words[25],
            "fixed_exec_offset": words[26],
            "exec_plan_offset": words[27],
            "window_sched_count": words[28],
            "window_cmd_count": words[29],
            "block5_sched_offset": words[30],
            "row_consumer_count": words[31],
        }
        if self.header["magic"] != PARAM_BLOB_MAGIC:
            raise ValueError(f"bad PARAM magic: 0x{self.header['magic']:08x}")
        if self.header["version"] != 4:
            raise ValueError(f"unsupported PARAM version: {self.header['version']}")

        # Read tensor descriptors. Section order defines tensor_id.
        self.tensor_desc: List[TensorDesc] = []
        td_count = self.header["tensor_desc_count"]
        td_off = self.header["tensor_desc_offset"]
        for i in range(td_count):
            bo = td_off + i * 16
            self.tensor_desc.append(TensorDesc(
                bank_id=self._u8(bo),
                elem_bytes=self._u8(bo + 1),
                reserved0=self._u16(bo + 2),
                base_offset=self._u32(bo + 4),
                h=self._u16(bo + 8),
                w=self._u16(bo + 10),
                c=self._u16(bo + 12),
                reserved1=self._u16(bo + 14),
            ))
        if len(self.tensor_desc) < 18:
            raise ValueError("PARAM v4 tensor descriptor table is incomplete")
        input_desc = self.tensor_desc[0]
        self.geometry = DeploymentGeometry(input_desc.h, input_desc.w, 8)
        logits_desc = self.tensor_desc[17]
        if (logits_desc.h, logits_desc.w) != (
            self.geometry.logits_height,
            self.geometry.logits_width,
        ):
            raise ValueError(
                "PARAM logits geometry disagrees with input geometry: "
                f"input={self.geometry.to_manifest()} "
                f"logits={(logits_desc.h, logits_desc.w)}"
            )
        self.class_count = int(logits_desc.c)
        if self.class_count not in (2, 20):
            raise ValueError(f"unsupported PARAM class_count={self.class_count}")
        # Read exec plan entries
        self.exec_plan: List[ExecPlanEntry] = []
        ep_count = self.header.get("exec_plan_count", 0)
        ep_off = self.header.get("exec_plan_offset", 0)
        for i in range(ep_count):
            bo = ep_off + i * 4
            self.exec_plan.append(ExecPlanEntry(
                kind=self._u8(bo), desc_id=self._u8(bo + 1),
                logical_uop_id=self._u8(bo + 2), flags=self._u8(bo + 3),
            ))
        # Read fixed exec descriptors. PARAM v4 places this section directly before exec_plan.
        self.fixed_exec: List[FixedExecDesc] = []
        fx_off = self.header.get("fixed_exec_offset", 0)
        if fx_off == 0 or ep_off < fx_off or (ep_off - fx_off) % 32 != 0:
            raise ValueError("bad fixed_exec_desc section")
        for i in range((ep_off - fx_off) // 32):
            bo = fx_off + i * 32
            self.fixed_exec.append(FixedExecDesc(
                kind=self._u8(bo),
                src0_tensor=self._u8(bo + 1),
                src1_tensor=self._u8(bo + 2),
                dst_tensor=self._u8(bo + 3),
                param_id=self._u8(bo + 4),
                add_param_id=self._u8(bo + 5),
                act_type=self._u8(bo + 6),
                flags=self._u8(bo + 7),
                in_h=self._u16(bo + 8),
                in_w=self._u16(bo + 10),
                in_c=self._u16(bo + 12),
                out_c=self._u16(bo + 14),
                kernel=self._u8(bo + 16),
                stride=self._u8(bo + 17),
                dilation=self._u8(bo + 18),
                padding=self._u8(bo + 19),
                c_offset=self._u16(bo + 20),
                valid_c=self._u16(bo + 22),
                qparam_id=self._u16(bo + 24),
                reserved0=self._u16(bo + 26),
                reserved1=self._u32(bo + 28),
            ))
        # Read fixed-size BLOCK5 schedule records. PARAM v4 stores the offset in header word 30.
        self.block5_sched: List[Block5SchedDesc] = []
        bs_off = self.header.get("block5_sched_offset", 0)
        if bs_off == 0:
            raise ValueError("PARAM v4 is missing block5_sched_desc section")
        for i in range(MAX_BLOCK5_SCHED_COUNT):
            bo = bs_off + i * 32
            self.block5_sched.append(
                Block5SchedDesc(
                    pattern=self._u8(bo),
                    branch_count=self._u8(bo + 1),
                    first_branch_conv_id=self._u8(bo + 2),
                    first_window_sched_id=self._u8(bo + 3),
                    first_conv_qparam_id=self._u8(bo + 4),
                    src_tensor=self._u8(bo + 5),
                    dst_tensor=self._u8(bo + 6),
                    add_tensor=self._u8(bo + 7),
                    chain_add_qparam_id0=self._u8(bo + 8),
                    chain_add_qparam_id1=self._u8(bo + 9),
                    chain_add_qparam_id2=self._u8(bo + 10),
                    residual_add_qparam_id=self._u8(bo + 11),
                    affine_param_id=self._u8(bo + 12),
                    affine_block_count=self._u8(bo + 13),
                    finalizer_kind=self._u8(bo + 14),
                    scratch_region=self._u8(bo + 15),
                    out_h=self._u16(bo + 16),
                    out_w=self._u16(bo + 18),
                    row_group_h=self._u16(bo + 20),
                    valid_c=self._u16(bo + 22),
                )
            )
        # Read conv exec descs
        self.conv_exec: Dict[int, ConvExecDesc] = {}
        ce_count = self.header.get("conv_exec_desc_count", 0)
        ce_off = self.header.get("conv_exec_desc_offset", 0)
        self.conv_exec_by_index: List[ConvExecDesc] = []
        for i in range(ce_count):
            bo = ce_off + i * 36
            desc = ConvExecDesc(
                param_id=self._u8(bo),
                qparam_id=self._u8(bo + 1),
                window_sched_id=self._u8(bo + 2),
                row_consumer_id=self._u8(bo + 3),
                in_h=self._u16(bo + 4), in_w=self._u16(bo + 6),
                in_c=self._u16(bo + 8), out_c=self._u16(bo + 10),
                kernel=self._u8(bo + 12), stride=self._u8(bo + 13),
                dilation=self._u8(bo + 14), padding=self._u8(bo + 15),
                src_tensor=self._u8(bo + 16), dst_tensor=self._u8(bo + 17),
                dst_c_offset=self._u16(bo + 18), valid_c=self._u16(bo + 20),
                packed_weight_word_offset=self._u32(bo + 22),
                k_tiles=self._u16(bo + 26), weight_words=self._u16(bo + 28),
                flags=self._u16(bo + 30),
                reserved=self._u16(bo + 32),
            )
            self.conv_exec[desc.param_id] = desc
            self.conv_exec_by_index.append(desc)
        # Read window sched descs
        self.window_sched: Dict[int, WindowSchedDesc] = {}
        ws_count = self.header.get("window_sched_count", 0)
        ws_off = self.header.get("window_sched_offset", 0)
        self.window_sched_by_index: List[WindowSchedDesc] = []
        for i in range(ws_count):
            bo = ws_off + i * 128
            kt_bases = []
            for k in range(41):
                kt_bases.append(self._u16(bo + 18 + k * 2))
            desc = WindowSchedDesc(
                mode=self._u8(bo), kernel=self._u8(bo + 1),
                stride=self._u8(bo + 2), dilation=self._u8(bo + 3),
                padding=self._u8(bo + 4), cache_chunks=self._u8(bo + 5),
                cache_col_slots=self._u8(bo + 6), flags=self._u8(bo + 7),
                in_c=self._u16(bo + 8), out_w=self._u16(bo + 10),
                k_tiles=self._u16(bo + 12),
                cmd_base=self._u16(bo + 14), cmd_count=self._u16(bo + 16),
                kt_cmd_base=kt_bases,
                loader_class=self._u8(bo + 100),
                loader_request_cols=self._u8(bo + 101),
                loader_warmup_issues=self._u8(bo + 102),
                loader_warmup_new_cols=self._u8(bo + 103),
                loader_steady_new_cols=self._u8(bo + 104),
                loader_words_per_col=self._u8(bo + 105),
                loader_warmup_mask=self._u8(bo + 106),
                loader_steady_mask=self._u8(bo + 107),
                reserved=[self._u16(bo + 108 + j * 2) for j in range(10)],
            )
            references = [conv for conv in self.conv_exec_by_index
                          if conv.window_sched_id == i]
            valid = window_row_reuse_contract_valid(desc)
            if desc.row_reuse_mode != WINDOW_ROW_REUSE_NONE:
                valid = bool(references) and all(
                    0 <= conv.src_tensor < len(self.tensor_desc)
                    and window_row_reuse_contract_valid(
                        desc, conv, self.tensor_desc[conv.src_tensor])
                    for conv in references)
            if not valid:
                raise ValueError(
                    f"window_sched[{i}] has invalid row-reuse contract: "
                    f"mode={desc.row_reuse_mode} "
                    f"words={desc.packed_words_per_source_row}"
                )
            self.window_sched[i] = desc
            self.window_sched_by_index.append(desc)
        # Read window pack commands
        self.pack_cmds: List[WindowPackCmd] = []
        wc_count = self.header.get("window_cmd_count", 0)
        wc_off = self.header.get("window_cmd_offset", 0)
        for i in range(wc_count):
            bo = wc_off + i * 8
            self.pack_cmds.append(WindowPackCmd(
                spatial_id=self._u8(bo), src_c_begin=self._u8(bo + 1),
                dst_lane_begin=self._u8(bo + 2), byte_count=self._u8(bo + 3),
                flags=self._u8(bo + 4),
            ))
        # Read row consumers
        self.row_consumer: Dict[int, RowConsumerDesc] = {}
        rc_count = self.header.get("row_consumer_count", 0)
        rc_off = self.header.get("row_consumer_offset", 0)
        for i in range(rc_count):
            bo = rc_off + i * 16
            self.row_consumer[i] = RowConsumerDesc(
                mode=self._u8(bo), add_other_tensor=self._u8(bo + 1),
                store_dst_tensor=self._u8(bo + 2), add_qparam_id=self._u8(bo + 3),
                store_c_offset=self._u16(bo + 4), valid_c=self._u16(bo + 6),
                alias_tensor=self._u8(bo + 8), affine_param_id=self._u8(bo + 9),
                affine_block_base=self._u8(bo + 10), act_type=self._u8(bo + 11),
                store_layout=self._u16(bo + 12),
                affine_c_offset=self._u16(bo + 14),
            )
        # Read conv qparams (param_id=0..25)
        self.conv_qparam: Dict[int, tuple] = {}
        cq_off = self.header.get("conv_qparam_offset", 0)
        for i in range(self.header["conv_exec_desc_count"]):
            bo = cq_off + i * 320
            bias = [self._i32(bo + j * 4) for j in range(32)]
            mult = [self._i32(bo + 128 + j * 4) for j in range(32)]
            shift = [self._u8(bo + 256 + j) for j in range(32)]
            self.conv_qparam[i] = (bias, mult, shift)
        # Read affine qparams. Each affine param may contain multiple 32-channel blocks.
        self.affine_qparam: Dict[int, List[tuple]] = {}
        aq_off = self.header.get("affine_qparam_offset", 0)
        affine_blocks = [1, 2, 2, 5, 4, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cursor = aq_off
        for pid in range(self.header["affine_desc_count"]):
            blocks = []
            for _ in range(affine_blocks[pid]):
                mul = [self._i32(cursor + j * 4) for j in range(32)]
                bias = [self._i32(cursor + 128 + j * 4) for j in range(32)]
                shift = [self._u8(cursor + 256 + j) for j in range(32)]
                blocks.append((mul, bias, shift))
                cursor += 320
            self.affine_qparam[pid] = blocks
        self.add_qparam: Dict[int, AddQParam] = {}
        add_off = self.header.get("add_qparam_offset", 0)
        for pid in range(self.header["add_desc_count"]):
            bo = add_off + pid * 24
            self.add_qparam[pid] = AddQParam(
                mult=self._i32(bo),
                shift=self._u8(bo + 4),
                act_type=self._u8(bo + 5),
                requant_bypass=self._u8(bo + 6),
                src_scale_id_a=self._u32(bo + 8),
                src_scale_id_b=self._u32(bo + 12),
                dst_scale_id=self._u32(bo + 16),
            )
        self.pool_qparam: Dict[int, PoolQParam] = {}
        pool_off = self.header.get("pool_qparam_offset", 0)
        for pid in range(self.header["pool_desc_count"]):
            bo = pool_off + pid * 28
            self.pool_qparam[pid] = PoolQParam(
                kernel=self._u8(bo),
                stride=self._u8(bo + 1),
                same_scale=self._u8(bo + 2),
                act_type=self._u8(bo + 3),
                src_scale_id=self._u32(bo + 4),
                dst_scale_id=self._u32(bo + 8),
                mult=self._i32(bo + 12),
                shift=self._u8(bo + 16),
            )
        # Read packed weights as raw int8 array
        pw_off = self.header.get("weight_packed_offset", 0)
        pw_end = self.header.get("conv_qparam_offset", len(self.data))
        # packed weights are stored as 32-byte words (256-bit wgt_vec_t)
        self.packed_weight_raw = self.data[pw_off:pw_end]

    def get_packed_weight_word(self, word_offset: int) -> np.ndarray:
        """Return 32 signed INT8 bytes for the packed weight word at offset."""
        bo = word_offset * 32
        return np.frombuffer(self.packed_weight_raw[bo:bo + 32], dtype=np.int8)

    def get_packed_weight_tile(self, param_id: int, oc: int, kt: int) -> np.ndarray:
        desc = self.conv_exec.get(param_id)
        if desc is None or oc >= desc.out_c or kt >= desc.k_tiles:
            return np.zeros(32, dtype=np.int8)
        word_offset = desc.packed_weight_word_offset + oc * desc.k_tiles + kt
        return self.get_packed_weight_word(word_offset)


# ── on-chip feature memory ─────────────────────────────────
class FeatureMemory:
    """Minimal on-chip FMBUF simulation for replay."""
    def __init__(self):
        self.buf = np.zeros(FMBUF_BYTES, dtype=np.uint8)

    @staticmethod
    def physical_base(desc: TensorDesc) -> int:
        if int(desc.bank_id) == BANK_BRAM_SCR0:
            return FMBUF_POOL_TMP_BASE + int(desc.base_offset)
        if int(desc.bank_id) == BANK_BRAM_SCR1:
            return FMBUF_POOL1_BASE + int(desc.base_offset)
        return int(desc.base_offset)

    def load_tile(self, desc: TensorDesc, h: int, w: int, c_begin: int, count: int) -> np.ndarray:
        if h < 0 or w < 0 or h >= desc.h or w >= desc.w or c_begin >= desc.c:
            return np.zeros(count, dtype=np.int8)
        valid = min(count, desc.c - c_begin, desc.phys_c)
        off = self.physical_base(desc) + desc.elem_offset(h, w, c_begin)
        result = np.zeros(count, dtype=np.int8)
        slice_buf = self.buf[off:off + valid].astype(np.int8)
        result[:valid] = slice_buf
        return result

    def store_tile(self, desc: TensorDesc, h: int, w: int, c_begin: int, count: int, data: np.ndarray):
        if h >= desc.h or w >= desc.w or c_begin >= desc.c:
            return
        valid = min(count, desc.c - c_begin)
        off = self.physical_base(desc) + desc.elem_offset(h, w, c_begin)
        self.buf[off:off + valid] = np.clip(data[:valid], INT8_MIN, INT8_MAX).astype(np.uint8)

    def dump_tensor(self, desc: TensorDesc) -> np.ndarray:
        result = np.zeros((desc.h, desc.w, desc.c), dtype=np.int8)
        for h in range(desc.h):
            for w in range(desc.w):
                off = self.physical_base(desc) + desc.elem_offset(h, w, 0)
                result[h, w, :] = self.buf[off:off + desc.c].astype(np.int8)
        return result


def load_dump_array(path: Path, desc: TensorDesc) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int8)
    expected = int(desc.h) * int(desc.w) * int(desc.c)
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} bytes, expected {expected} for {desc.h}x{desc.w}x{desc.c}")
    return data.reshape((int(desc.h), int(desc.w), int(desc.c)))


def load_dump_into_fmem(fmem: Optional[FeatureMemory], desc: TensorDesc, path: Path) -> np.ndarray:
    arr = load_dump_array(path, desc)
    if fmem is None:
        return arr
    for h in range(desc.h):
        for w in range(desc.w):
            fmem.store_tile(desc, h, w, 0, desc.c, arr[h, w, :])
    return arr


def compare_i8_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = a.astype(np.int16) - b.astype(np.int16)
    abs_diff = np.abs(diff)
    mismatches = int((diff != 0).sum())
    return {
        "shape": list(a.shape),
        "bytes": int(a.size),
        "mismatches": mismatches,
        "mismatch_rate": float(mismatches / a.size) if a.size else 0.0,
        "max_abs_diff": int(abs_diff.max()) if a.size else 0,
        "mean_abs_diff": float(abs_diff.mean()) if a.size else 0.0,
    }


def _write_array_to_fmem(fmem: FeatureMemory, desc: TensorDesc, arr: np.ndarray) -> None:
    if arr.shape != (int(desc.h), int(desc.w), int(desc.c)):
        raise ValueError(f"shape mismatch for fmem write: {arr.shape} vs {desc.h}x{desc.w}x{desc.c}")
    if desc.c_offset == 0 and desc.phys_c == desc.c:
        flat = np.ascontiguousarray(arr.reshape(-1)).astype(np.uint8)
        start = fmem.physical_base(desc)
        fmem.buf[start:start + flat.size] = flat
        return
    for h in range(desc.h):
        for w in range(desc.w):
            fmem.store_tile(desc, h, w, 0, desc.c, arr[h, w, :])


def _store_slice(dst: Optional[np.ndarray], dst_shape: Tuple[int, int, int], c_offset: int, src: np.ndarray, valid_c: int = 0) -> np.ndarray:
    if dst is None:
        dst = np.zeros(dst_shape, dtype=np.int8)
    elif dst.shape[0] != dst_shape[0] or dst.shape[1] != dst_shape[1] or dst.shape[2] != dst_shape[2]:
        # PARAM replay is array-level, while the hardware schedule reuses a few
        # logical scratch tensor ids across stages with different H/W/C shapes.
        # A shape change means a new scratch lifetime, not an in-place update.
        dst = np.zeros(dst_shape, dtype=np.int8)
    count = int(valid_c) if valid_c else int(src.shape[2])
    if dst.shape[0] != src.shape[0] or dst.shape[1] != src.shape[1]:
        raise ValueError(f"store spatial mismatch: dst={dst.shape}, src={src.shape}")
    if c_offset + count > dst.shape[2] or count > src.shape[2]:
        raise ValueError(f"store channel mismatch: dst={dst.shape}, src={src.shape}, c_offset={c_offset}, count={count}")
    dst[:, :, c_offset:c_offset + count] = src[:, :, :count]
    return dst


def _tensor_shape(blob: ParamBlob, tid: int, fallback: Optional[Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
    if 0 <= tid < len(blob.tensor_desc):
        desc = blob.tensor_desc[tid]
        return (int(desc.h), int(desc.w), int(desc.c))
    if fallback is not None:
        return fallback
    raise KeyError(f"no tensor descriptor for tensor_id={tid}")


def replay_conv_array(blob: ParamBlob, param_id: int, src: np.ndarray) -> np.ndarray:
    desc = blob.conv_exec.get(param_id)
    if desc is None:
        raise KeyError(f"conv exec desc not found for param_id={param_id}")
    expected = (int(desc.in_h), int(desc.in_w), int(desc.in_c))
    if src.shape != expected:
        raise ValueError(f"conv{param_id} source shape mismatch: {src.shape} vs {expected}")

    stride_val = desc.stride if desc.stride else 1
    out_h = math.ceil(desc.in_h / stride_val)
    out_w = math.ceil(desc.in_w / stride_val)
    out_c = int(desc.out_c)

    fmem = FeatureMemory()
    src_desc = TensorDesc(base_offset=0, h=desc.in_h, w=desc.in_w, c=desc.in_c, reserved0=desc.in_c)
    src_bytes = int(desc.in_h) * int(desc.in_w) * int(desc.in_c)
    dst_base = ((src_bytes + 63) // 64) * 64
    dst_desc = TensorDesc(base_offset=dst_base, h=out_h, w=out_w, c=out_c, reserved0=out_c)
    _write_array_to_fmem(fmem, src_desc, src)
    replay_conv(fmem, blob, param_id, src_desc, dst_desc, RowConsumerDesc())
    return fmem.dump_tensor(dst_desc)


def replay_affine_array(blob: ParamBlob,
                        param_id: int,
                        src: np.ndarray,
                        act_type: int = ACT_NONE,
                        c_offset: int = 0) -> np.ndarray:
    blocks = blob.affine_qparam.get(param_id)
    if not blocks:
        raise KeyError(f"affine qparam not found for param_id={param_id}")
    dst = np.zeros_like(src, dtype=np.int8)
    for ch in range(src.shape[2]):
        absolute_ch = int(c_offset) + ch
        block_idx = absolute_ch // 32
        lane = absolute_ch % 32
        if block_idx >= len(blocks):
            raise ValueError(f"affine param_id={param_id} missing block {block_idx} for channel {ch}")
        mul, bias, shift = blocks[block_idx]
        dst[:, :, ch] = affine_i8_to_i8_np(src[:, :, ch], mul[lane], bias[lane], shift[lane], act_type)
    return dst


def _read_block5_sources(tensors: Dict[int, np.ndarray], fixed: FixedExecDesc, logical: int) -> List[np.ndarray]:
    src2, src3, src4, add_tensor = unpack_block5_reserved1(fixed.reserved1)
    if add_tensor != BLOCK5_ADD_TENSOR_NONE:
        raise ValueError(
            f"BLOCK5 source unpack logical={logical} unexpectedly carries add tensor {add_tensor}; "
            "use replay_block5_add_affine for add-affine descriptors"
        )
    parts: List[np.ndarray] = []
    for tid in (fixed.src0_tensor, fixed.src1_tensor, src2, src3, src4):
        src = tensors.get(tid)
        if src is None:
            raise KeyError(f"BLOCK5 logical={logical} missing source tensor {tid}")
        parts.append(src)
    base_hw = parts[0].shape[:2]
    for src in parts[1:]:
        if src.shape[:2] != base_hw:
            raise ValueError(f"BLOCK5 logical={logical} spatial mismatch: {[p.shape for p in parts]}")
    return parts


def replay_block5_affine(tensors: Dict[int, np.ndarray], blob: ParamBlob, fixed: FixedExecDesc, logical: int) -> np.ndarray:
    parts = _read_block5_sources(tensors, fixed, logical)
    cat = np.concatenate(parts, axis=2)
    expected_channels = block5_source_channels(int(fixed.reserved0) & 0xFF) if fixed.reserved0 else None
    if expected_channels is not None and [p.shape[2] for p in parts] != list(expected_channels):
        raise ValueError(
            f"BLOCK5_AFFINE logical={logical} source channel mismatch: "
            f"{[p.shape[2] for p in parts]} vs {list(expected_channels)}"
        )
    if int(fixed.valid_c) and cat.shape[2] != int(fixed.valid_c):
        raise ValueError(f"BLOCK5_AFFINE logical={logical} channel mismatch: cat={cat.shape}, fixed={fixed}")
    return replay_affine_array(blob, int(fixed.param_id), cat, fixed.act_type)


def replay_block5_add_affine(tensors: Dict[int, np.ndarray], blob: ParamBlob, fixed: FixedExecDesc, logical: int) -> np.ndarray:
    src2, src3, src4, add_tensor = unpack_block5_reserved1(fixed.reserved1)
    parts: List[np.ndarray] = []
    for tid in (fixed.src0_tensor, fixed.src1_tensor, src2, src3, src4):
        src = tensors.get(tid)
        if src is None:
            raise KeyError(f"BLOCK5_ADD_AFFINE logical={logical} missing source tensor {tid}")
        parts.append(src)
    expected_channels = block5_source_channels(int(fixed.reserved0) & 0xFF) if fixed.reserved0 else None
    if expected_channels is not None and [p.shape[2] for p in parts] != list(expected_channels):
        raise ValueError(
            f"BLOCK5_ADD_AFFINE logical={logical} source channel mismatch: "
            f"{[p.shape[2] for p in parts]} vs {list(expected_channels)}"
        )
    cat = np.concatenate(parts, axis=2)
    residual = tensors.get(add_tensor)
    if residual is None:
        raise KeyError(f"BLOCK5_ADD_AFFINE logical={logical} missing residual tensor {add_tensor}")
    if cat.shape != residual.shape:
        raise ValueError(
            f"BLOCK5_ADD_AFFINE logical={logical} shape mismatch: cat={cat.shape}, residual={residual.shape}"
        )
    if int(fixed.valid_c) and cat.shape[2] != int(fixed.valid_c):
        raise ValueError(f"BLOCK5_ADD_AFFINE logical={logical} channel mismatch: cat={cat.shape}, fixed={fixed}")
    added = add_i8_array(cat, residual, blob.add_qparam[fixed.add_param_id])
    return replay_affine_array(blob, int(fixed.param_id), added, fixed.act_type)


def replay_block5_row_group(tensors: Dict[int, np.ndarray], blob: ParamBlob, fixed: FixedExecDesc, logical: int) -> np.ndarray:
    sched_id = (int(fixed.reserved0) >> 8) & 0xFF
    if sched_id >= len(blob.block5_sched):
        raise ValueError(f"BLOCK5 row-group logical={logical} bad sched_id={sched_id}")
    bs = blob.block5_sched[sched_id]
    pattern = int(fixed.reserved0) & 0xFF
    if bs.pattern != pattern:
        raise ValueError(f"BLOCK5 row-group logical={logical} pattern mismatch: fixed={pattern} schedule={bs.pattern}")
    if bs.branch_count != 5:
        raise ValueError(f"BLOCK5 row-group logical={logical} branch_count={bs.branch_count}, expected 5")
    base_desc = int(bs.first_branch_conv_id)
    if base_desc < 0 or base_desc + 4 >= len(blob.conv_exec_by_index):
        raise ValueError(f"BLOCK5 row-group logical={logical} bad base conv desc id {base_desc}")
    src = tensors.get(bs.src_tensor)
    if src is None:
        raise KeyError(f"BLOCK5 row-group logical={logical} missing source tensor {bs.src_tensor}")

    raw: List[np.ndarray] = []
    for offset in range(5):
        conv = blob.conv_exec_by_index[base_desc + offset]
        if conv.src_tensor != bs.src_tensor:
            raise ValueError(
                f"BLOCK5 row-group logical={logical} branch {offset} source mismatch: "
                f"{conv.src_tensor} vs {bs.src_tensor}"
            )
        raw.append(replay_conv_array(blob, int(conv.param_id), src))

    expected_channels = block5_source_channels(pattern)
    if [x.shape[2] for x in raw] != list(expected_channels):
        raise ValueError(
            f"BLOCK5 row-group logical={logical} raw channel mismatch: "
            f"{[x.shape[2] for x in raw]} vs {list(expected_channels)}"
        )
    branch1 = raw[1]
    chain0 = add_i8_array(branch1, raw[2], blob.add_qparam[int(bs.chain_add_qparam_id0)])
    chain1 = add_i8_array(chain0, raw[3], blob.add_qparam[int(bs.chain_add_qparam_id1)])
    chain2 = add_i8_array(chain1, raw[4], blob.add_qparam[int(bs.chain_add_qparam_id2)])
    cat = np.concatenate([raw[0], branch1, chain0, chain1, chain2], axis=2)
    if int(bs.valid_c) and cat.shape[2] != int(bs.valid_c):
        raise ValueError(f"BLOCK5 row-group logical={logical} concat channel mismatch: cat={cat.shape}, fixed={fixed}")
    if bs.add_tensor != BLOCK5_ADD_TENSOR_NONE:
        residual = tensors.get(bs.add_tensor)
        if residual is None:
            raise KeyError(f"BLOCK5 row-group logical={logical} missing residual tensor {bs.add_tensor}")
        if residual.shape != cat.shape:
            raise ValueError(
                f"BLOCK5 row-group logical={logical} residual shape mismatch: cat={cat.shape}, residual={residual.shape}"
            )
        cat = add_i8_array(cat, residual, blob.add_qparam[int(bs.residual_add_qparam_id)])
    return replay_affine_array(blob, int(bs.affine_param_id), cat, fixed.act_type)


def _trunc_div(num: int, den: int) -> int:
    return int(num / den)


def _round_div9_np(x: np.ndarray) -> np.ndarray:
    positive = (x + 4) // 9
    # C++ signed integer division truncates toward zero, unlike Python //.
    negative = np.vectorize(lambda v: _trunc_div(int(v) - 4, 9), otypes=[np.int32])(x)
    return np.where(x >= 0, positive, negative).astype(np.int32)


def replay_pool_array(blob: ParamBlob, param_id: int, src: np.ndarray) -> np.ndarray:
    q = blob.pool_qparam[param_id]
    kernel = int(q.kernel) if q.kernel else 3
    stride = int(q.stride) if q.stride else 2
    if kernel != 3 or stride != 2 or src.shape[2] != 3:
        raise ValueError(f"unsupported pool shape: param={param_id}, kernel={kernel}, stride={stride}, src={src.shape}")
    out_h = math.ceil(src.shape[0] / stride)
    out_w = math.ceil(src.shape[1] / stride)
    dst = np.zeros((out_h, out_w, src.shape[2]), dtype=np.int8)
    for oh in range(out_h):
        for ow in range(out_w):
            sums = np.zeros(src.shape[2], dtype=np.int32)
            for kh in range(3):
                ih = oh * 2 + kh - 1
                if ih < 0 or ih >= src.shape[0]:
                    continue
                for kw in range(3):
                    iw = ow * 2 + kw - 1
                    if iw < 0 or iw >= src.shape[1]:
                        continue
                    sums += src[ih, iw, :].astype(np.int32)
            avg = _round_div9_np(sums)
            if q.same_scale:
                out = np.clip(avg, INT8_MIN, INT8_MAX).astype(np.int8)
                if q.act_type == ACT_RELU:
                    out = np.clip(out, 0, None).astype(np.int8)
            else:
                out = affine_i8_to_i8_np(avg.astype(np.int8), q.mult, 0, q.shift, q.act_type)
            dst[oh, ow, :] = out
    return dst


def add_i8_array(a: np.ndarray, b: np.ndarray, q: AddQParam) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"add shape mismatch: {a.shape} vs {b.shape}")
    summed = a.astype(np.int32) + b.astype(np.int32)
    if q.requant_bypass:
        out = np.clip(summed, INT8_MIN, INT8_MAX).astype(np.int8)
        if q.act_type == ACT_RELU:
            out = np.clip(out, 0, None).astype(np.int8)
        return out
    scaled = summed.astype(np.int64) * np.int64(q.mult)
    rounded = np.vectorize(lambda v: hls_round_shift_int(int(v), int(q.shift)), otypes=[np.int32])(scaled)
    out = np.clip(rounded, INT8_MIN, INT8_MAX).astype(np.int8)
    if q.act_type == ACT_RELU:
        out = np.clip(out, 0, None).astype(np.int8)
    return out


@dataclass
class PrefixReplayResult:
    tensors: Dict[int, np.ndarray]
    executed: List[int]

    def read_tensor(self, tid: int) -> np.ndarray:
        if tid not in self.tensors:
            raise KeyError(f"tensor_id={tid} was not produced by prefix replay")
        return self.tensors[tid]


def replay_prefix(blob: ParamBlob, input_bin: Path, stop_logical_uop: int) -> PrefixReplayResult:
    """Closed-loop PARAM v4 replay through ``stop_logical_uop``.

    This follows the compiled exec_plan and row-consumer descriptors. It is
    intentionally array-level: the goal is to isolate fixed-point execution
    semantics from the physical memory banking policy.
    """
    from export_int8_hw_blob import TID_INVALID

    tensors: Dict[int, np.ndarray] = {}
    executed: List[int] = []

    input_data = np.fromfile(input_bin, dtype=np.int8)
    input_desc = blob.tensor_desc[0]
    expected = int(input_desc.h) * int(input_desc.w) * int(input_desc.c)
    if input_data.size != expected:
        raise ValueError(f"{input_bin} has {input_data.size} bytes, expected {expected}")
    tensors[0] = input_data.reshape((int(input_desc.h), int(input_desc.w), int(input_desc.c)))

    def store_to_tensor(dst_tid: int, src: np.ndarray, c_offset: int = 0, valid_c: int = 0) -> None:
        if dst_tid == TID_INVALID:
            return
        shape = _tensor_shape(blob, dst_tid, fallback=src.shape)
        existing = tensors.get(dst_tid)
        tensors[dst_tid] = _store_slice(existing, shape, int(c_offset), src, int(valid_c) if valid_c else src.shape[2])

    for entry in blob.exec_plan:
        if entry.kind == EXEC_END:
            break
        logical = int(entry.logical_uop_id)
        if logical > stop_logical_uop:
            break

        if entry.kind == EXEC_CONV:
            desc = blob.conv_exec_by_index[int(entry.desc_id)]
            src = tensors.get(desc.src_tensor)
            if src is None:
                raise KeyError(f"conv param_id={desc.param_id} missing src tensor {desc.src_tensor}")
            out = replay_conv_array(blob, desc.param_id, src)
            consumer = blob.row_consumer.get(desc.row_consumer_id, RowConsumerDesc())
            if consumer.mode == ROW_CONSUMER_NONE:
                store_to_tensor(desc.dst_tensor, out, desc.dst_c_offset, desc.valid_c)
            elif consumer.mode == ROW_CONSUMER_STORE:
                store_to_tensor(consumer.store_dst_tensor, out, consumer.store_c_offset, consumer.valid_c)
                if consumer.alias_tensor != TID_INVALID:
                    tensors[consumer.alias_tensor] = out
            elif consumer.mode == ROW_CONSUMER_ADD_STORE:
                other = tensors.get(consumer.add_other_tensor)
                if other is None:
                    raise KeyError(f"conv param_id={desc.param_id} missing add tensor {consumer.add_other_tensor}")
                added = add_i8_array(out, other, blob.add_qparam[consumer.add_qparam_id])
                store_to_tensor(consumer.store_dst_tensor, added, consumer.store_c_offset, consumer.valid_c)
                if consumer.alias_tensor != TID_INVALID:
                    tensors[consumer.alias_tensor] = added
            elif consumer.mode == ROW_CONSUMER_ADD_STORE_BLOCK_ADD_AFFINE:
                other = tensors.get(consumer.add_other_tensor)
                if other is None:
                    raise KeyError(f"conv param_id={desc.param_id} missing add tensor {consumer.add_other_tensor}")
                added = add_i8_array(out, other, blob.add_qparam[consumer.add_qparam_id])
                store_to_tensor(consumer.store_dst_tensor, added, consumer.store_c_offset, consumer.valid_c)
                cat = tensors.get(consumer.store_dst_tensor)
                residual = None if consumer.affine_c_offset == TID_INVALID else tensors.get(consumer.affine_c_offset)
                if cat is None or (consumer.affine_c_offset != TID_INVALID and residual is None):
                    raise KeyError(
                        f"conv param_id={desc.param_id} missing block tensors "
                        f"{consumer.store_dst_tensor}/{consumer.affine_c_offset}"
                    )
                block_added = (
                    add_i8_array(cat, residual, blob.add_qparam[consumer.affine_block_base])
                    if residual is not None
                    else cat
                )
                affined = replay_affine_array(blob, consumer.affine_param_id, block_added, consumer.act_type)
                tensors[consumer.alias_tensor] = affined
            elif consumer.mode == ROW_CONSUMER_AFFINE_STORE:
                affined = replay_affine_array(blob,
                                              consumer.affine_param_id,
                                              out,
                                              consumer.act_type,
                                              consumer.affine_c_offset)
                store_to_tensor(consumer.store_dst_tensor, affined, consumer.store_c_offset, consumer.valid_c)
                if consumer.alias_tensor != TID_INVALID:
                    tensors[consumer.alias_tensor] = affined
            elif consumer.mode == ROW_CONSUMER_ADD_AFFINE_STORE:
                other = tensors.get(consumer.add_other_tensor)
                if other is None:
                    raise KeyError(f"conv param_id={desc.param_id} missing add tensor {consumer.add_other_tensor}")
                added = add_i8_array(out, other, blob.add_qparam[consumer.add_qparam_id])
                affined = replay_affine_array(blob,
                                              consumer.affine_param_id,
                                              added,
                                              consumer.act_type,
                                              consumer.affine_c_offset)
                store_to_tensor(consumer.store_dst_tensor, affined, consumer.store_c_offset, consumer.valid_c)
                if consumer.alias_tensor != TID_INVALID:
                    tensors[consumer.alias_tensor] = affined
            elif consumer.mode == ROW_CONSUMER_CAT_AFFINE_STORE:
                tail = tensors.get(consumer.add_other_tensor)
                if tail is None:
                    raise KeyError(f"conv param_id={desc.param_id} missing cat tensor {consumer.add_other_tensor}")
                needed_tail = int(consumer.valid_c) - out.shape[2]
                if needed_tail <= 0:
                    raise ValueError(f"conv param_id={desc.param_id} invalid cat width: out={out.shape} consumer={consumer}")
                cat = np.concatenate([out, tail[:, :, :needed_tail]], axis=2)
                affined = replay_affine_array(blob,
                                              consumer.affine_param_id,
                                              cat,
                                              consumer.act_type,
                                              consumer.affine_c_offset)
                store_to_tensor(consumer.store_dst_tensor, affined, consumer.store_c_offset, consumer.valid_c)
            elif consumer.mode == ROW_CONSUMER_UPSAMPLE_OUT:
                tensors[desc.dst_tensor] = out
            else:
                raise ValueError(f"unsupported row consumer mode={consumer.mode}")
            executed.append(logical)
            continue

        if entry.kind == EXEC_POOL:
            fixed = blob.fixed_exec[int(entry.desc_id)]
            if fixed.kind != entry.kind:
                raise ValueError(f"exec pool desc kind mismatch: {fixed.kind} != {entry.kind}")
            src = tensors.get(fixed.src0_tensor)
            if src is None:
                raise KeyError(f"pool param_id={fixed.param_id} missing src tensor {fixed.src0_tensor}")
            tensors[fixed.dst_tensor] = replay_pool_array(blob, int(fixed.param_id), src)
            executed.append(logical)
            continue

        if entry.kind in REMOVED_P6_EXEC_KINDS:
            raise ValueError(
                f"removed P6 fixed-op kind={entry.kind} reached replay at logical={logical}; "
                "PARAM v4 must fuse affine/store/add-affine into row consumers or block fixed ops"
            )

        if entry.kind == EXEC_BLOCK_AFFINE:
            fixed = blob.fixed_exec[int(entry.desc_id)]
            if fixed.kind != entry.kind:
                raise ValueError(f"exec block-affine desc kind mismatch: {fixed.kind} != {entry.kind}")
            if fixed.flags & FIXED_FLAG_BLOCK5_ROW_GROUP:
                tensors[fixed.dst_tensor] = replay_block5_row_group(tensors, blob, fixed, logical)
                executed.append(logical)
                continue
            if fixed.flags & FIXED_FLAG_BLOCK5_AFFINE:
                tensors[fixed.dst_tensor] = replay_block5_affine(tensors, blob, fixed, logical)
                executed.append(logical)
                continue
            parts = []
            for tid in (fixed.src0_tensor, fixed.src1_tensor, fixed.reserved0):
                if tid == TID_INVALID:
                    continue
                src = tensors.get(tid)
                if src is None:
                    raise KeyError(f"block-affine logical={logical} missing src tensor {tid}")
                parts.append(src)
            if not parts:
                raise ValueError(f"block-affine logical={logical} has no source tensors")
            cat = np.concatenate(parts, axis=2)
            if int(fixed.valid_c) and cat.shape[2] != int(fixed.valid_c):
                raise ValueError(f"block-affine channel mismatch logical={logical}: cat={cat.shape}, fixed={fixed}")
            tensors[fixed.dst_tensor] = replay_affine_array(blob, int(fixed.param_id), cat, fixed.act_type)
            executed.append(logical)
            continue

        if entry.kind == EXEC_BLOCK_ADD_AFFINE:
            fixed = blob.fixed_exec[int(entry.desc_id)]
            if fixed.kind != entry.kind:
                raise ValueError(f"exec block-add-affine desc kind mismatch: {fixed.kind} != {entry.kind}")
            if fixed.flags & FIXED_FLAG_BLOCK5_ROW_GROUP:
                tensors[fixed.dst_tensor] = replay_block5_row_group(tensors, blob, fixed, logical)
                executed.append(logical)
                continue
            if fixed.flags & FIXED_FLAG_BLOCK5_ADD_AFFINE:
                tensors[fixed.dst_tensor] = replay_block5_add_affine(tensors, blob, fixed, logical)
                executed.append(logical)
                continue
            parts = []
            for tid in (fixed.src0_tensor, fixed.src1_tensor):
                src = tensors.get(tid)
                if src is None:
                    raise KeyError(f"block-add-affine logical={logical} missing cat src tensor {tid}")
                parts.append(src)
            cat = np.concatenate(parts, axis=2)
            residual = tensors.get(fixed.reserved0)
            if residual is None:
                raise KeyError(f"block-add-affine logical={logical} missing residual tensor {fixed.reserved0}")
            if cat.shape != residual.shape:
                raise ValueError(
                    f"block-add-affine shape mismatch logical={logical}: cat={cat.shape}, residual={residual.shape}"
                )
            if int(fixed.valid_c) and cat.shape[2] != int(fixed.valid_c):
                raise ValueError(f"block-add-affine channel mismatch logical={logical}: cat={cat.shape}, fixed={fixed}")
            added = add_i8_array(cat, residual, blob.add_qparam[fixed.add_param_id])
            tensors[fixed.dst_tensor] = replay_affine_array(blob, int(fixed.param_id), added, fixed.act_type)
            executed.append(logical)
            continue

        raise ValueError(f"unsupported exec kind={entry.kind} at logical={logical}")

    return PrefixReplayResult(tensors=tensors, executed=executed)


# ── conv replay ─────────────────────────────────────────────
def linear_k_to_spatial_c(k: int, in_c: int, kernel: int) -> Tuple[int, int, int, int]:
    spatial = k // in_c
    cin = k % in_c
    kh = spatial // kernel
    kw = spatial % kernel
    return spatial, kh, kw, cin


def requant_i32_to_i8(acc, bias, mult, shift, act_type):
    """Mirror HLS requant_i32_to_i8."""
    biased = np.int64(acc) + np.int64(bias)
    scaled = biased * np.int64(mult)
    shift_val = int(shift)
    if shift_val > 0:
        bias_r = np.int64(1) << (shift_val - 1)
        scale = np.int64(1) << shift_val
        rounded = np.where(scaled >= 0, (scaled + bias_r) >> shift_val,
                          ((scaled - bias_r) + scale - 1) >> shift_val)
    else:
        rounded = scaled
    out = np.clip(rounded, INT8_MIN, INT8_MAX).astype(np.int8)
    if act_type == ACT_RELU:
        out = np.clip(out, 0, None)
    return out


def affine_i8_to_i8_np(x: np.ndarray, mul: int, bias: int, shift: int, act_type: int) -> np.ndarray:
    scaled = x.astype(np.int64) * np.int64(mul) + np.int64(bias)
    shift_val = int(shift)
    if shift_val > 0:
        bias_r = np.int64(1) << (shift_val - 1)
        scale = np.int64(1) << shift_val
        rounded = np.where(scaled >= 0, (scaled + bias_r) >> shift_val,
                           ((scaled - bias_r) + scale - 1) >> shift_val)
    else:
        rounded = scaled
    out = np.clip(rounded, INT8_MIN, INT8_MAX).astype(np.int8)
    if act_type == ACT_RELU:
        out = np.clip(out, 0, None).astype(np.int8)
    return out


def replay_affine_tensor(
    fmem: FeatureMemory,
    blob: ParamBlob,
    param_id: int,
    src_desc: TensorDesc,
    dst_desc: TensorDesc,
    act_type: int = ACT_NONE,
) -> None:
    blocks = blob.affine_qparam.get(param_id)
    if not blocks:
        raise KeyError(f"affine qparam not found for param_id={param_id}")
    src = fmem.dump_tensor(src_desc)
    dst = np.zeros((dst_desc.h, dst_desc.w, dst_desc.c), dtype=np.int8)
    for ch in range(dst_desc.c):
        block_idx = ch // 32
        lane = ch % 32
        if block_idx >= len(blocks):
            raise ValueError(f"affine param_id={param_id} missing block {block_idx} for channel {ch}")
        mul, bias, shift = blocks[block_idx]
        dst[:, :, ch] = affine_i8_to_i8_np(src[:, :, ch], mul[lane], bias[lane], shift[lane], act_type)
    for h in range(dst_desc.h):
        for w in range(dst_desc.w):
            fmem.store_tile(dst_desc, h, w, 0, dst_desc.c, dst[h, w, :])


def replay_conv(fmem: FeatureMemory, blob: ParamBlob, param_id: int, src_desc: TensorDesc, dst_desc: TensorDesc, consumer: RowConsumerDesc) -> None:
    desc = blob.conv_exec.get(param_id)
    if desc is None:
        raise KeyError(f"conv exec desc not found for param_id={param_id}")
    sched_id = desc.window_sched_id
    sched = blob.window_sched.get(sched_id)
    bias, mult, shift = blob.conv_qparam.get(desc.qparam_id, ([0] * 32, [0] * 32, [0] * 32))

    stride_val = sched.stride if sched and sched.stride else (desc.stride if desc.stride else 1)
    dilation_val = sched.dilation if sched and sched.dilation else (desc.dilation if desc.dilation else 1)
    padding_val = sched.padding if sched is not None else desc.padding
    out_h = math.ceil(desc.in_h / stride_val)
    out_w = sched.out_w if sched and sched.out_w else math.ceil(desc.in_w / stride_val)
    kernel_val = sched.kernel if sched and sched.kernel else desc.kernel
    in_c = sched.in_c if sched and sched.in_c else desc.in_c
    out_c = desc.out_c
    k_tiles = sched.k_tiles if sched and sched.k_tiles else desc.k_tiles
    act_type = ACT_RELU if (desc.flags & FLAG_RELU_EN) else ACT_NONE

    for oh in range(out_h):
        for ow in range(out_w):
            # Build activation words for all K tiles
            act_words = []
            for kt in range(k_tiles):
                word = np.zeros(TK, dtype=np.int8)
                # P7 HLS WinGen no longer executes window_pack_cmd at runtime.
                # Replay uses the same linear-K ordering as the staged hardware
                # builders and the SA-ready weight packer.
                for lane in range(TK):
                    k = kt * TK + lane
                    k_total = kernel_val * kernel_val * in_c
                    if k >= k_total:
                        break
                    spatial, kh, kw, cin = linear_k_to_spatial_c(k, in_c, kernel_val)
                    ih = oh * stride_val + kh * dilation_val - padding_val
                    iw = ow * stride_val + kw * dilation_val - padding_val
                    tile = fmem.load_tile(src_desc, ih, iw, cin, 1)
                    word[lane] = tile[0]
                act_words.append(word)

            # SA: for each OC within TM tile, accumulate over KT
            c_offset = desc.dst_c_offset
            out_tile = np.zeros(out_c, dtype=np.int8)
            for oc in range(out_c):
                psum = np.int32(0)
                for kt in range(k_tiles):
                    wgt_vec = blob.get_packed_weight_tile(param_id, oc, kt)
                    act_vec = act_words[kt]
                    psum += np.dot(act_vec.astype(np.int32), wgt_vec.astype(np.int32))
                out_tile[oc] = requant_i32_to_i8(psum, bias[oc], mult[oc], shift[oc], act_type)
            fmem.store_tile(dst_desc, oh, ow, c_offset, out_c, out_tile)


# ── main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Offline P7 PARAM replay")
    parser.add_argument("--param", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="Optional legacy argument retained for command compatibility",
    )
    parser.add_argument("--layer", type=str, default="all",
                        help="Layer to replay: all, U38, U39, U40")
    parser.add_argument(
        "--stop-logical-uop",
        type=int,
        default=None,
        help="Stop after this PARAM logical UOP; overrides --layer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the replayed T_OUT tensor in HWC int8 order",
    )
    args = parser.parse_args()

    blob = ParamBlob(args.param)
    input_desc = blob.tensor_desc[0]
    input_data = np.fromfile(args.input, dtype=np.int8)
    expected_input = int(input_desc.h) * int(input_desc.w) * int(input_desc.c)
    if input_data.size != expected_input:
        raise ValueError(f"{args.input} has {input_data.size} bytes, expected {expected_input}")

    if args.stop_logical_uop is not None:
        stop = int(args.stop_logical_uop)
    elif args.layer.lower() == "all":
        stop = max(
            (int(entry.logical_uop_id) for entry in blob.exec_plan if entry.kind != EXEC_END),
            default=0,
        )
    elif args.layer.upper().startswith("U") and args.layer[1:].isdigit():
        stop = int(args.layer[1:])
    else:
        raise ValueError(f"unsupported replay layer selector: {args.layer}")

    print(
        f"PARAM: geometry={blob.geometry.to_manifest()} classes={blob.class_count} "
        f"exec_plan={len(blob.exec_plan)} conv_exec={len(blob.conv_exec)} stop_uop={stop}"
    )
    replay = replay_prefix(blob, args.input, stop)
    print(f"Replay complete: executed={len(replay.executed)} logical_uops={replay.executed}")

    if 17 in replay.tensors:
        output = replay.tensors[17]
        print(f"T_OUT: shape={output.shape} bytes={output.nbytes}")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.ascontiguousarray(output, dtype=np.int8).tofile(args.output)
            print(f"Saved T_OUT: {args.output}")
    elif args.output is not None:
        raise ValueError("requested --output, but replay stop point did not produce T_OUT")


if __name__ == "__main__":
    main()
