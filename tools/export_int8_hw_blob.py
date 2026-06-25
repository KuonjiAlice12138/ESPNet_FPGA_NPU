#!/usr/bin/env python3
"""
Export ESPNet INT8 artifacts into the current HLS/NPU binary contract.

This script consumes the existing model-side artifact directory produced by
D:\\ESPNet\\export_quantized_artifacts.py and emits the hardware-facing files
under D:\\ESP_INT8 by default. The hardware-facing contract includes the
shared-FMBUF memory layout, tensor view descriptors, and uop schedule used by
the current ESP_INT8_hls implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


PARAM_BLOB_MAGIC = 0x544E4945
PARAM_BLOB_VERSION = 0x00010000
PARAM_BLOB_VERSION_SCHED = 3
SECTION_ALIGN = 64
PARAM_HEADER_WORDS = 32
PARAM_HEADER_BYTES = PARAM_HEADER_WORDS * 4

TM = 32
TK = 32
WBUF_BYTES = 128 * 1024
MAX_K_TILE_COUNT = 40
MAX_PACK_CMDS_PER_KT = 9
MAX_WINDOW_PACK_CMD_COUNT = 2048
MAX_EXEC_PLAN_COUNT = 96

ACT_NONE = 0
ACT_RELU = 1

UOP_NOP = 0
UOP_LOAD_FM = 1
UOP_CONV = 2
UOP_POOL = 3
UOP_ADD = 4
UOP_AFFINE = 5
UOP_STORE = 6
UOP_END = 15

WIN_MODE_INVALID = 0
WIN_MODE_SMALLC_3X3_STAGED = 1
WIN_MODE_LARGEC_3X3_SEGMENT = 2
WIN_MODE_1X1_ALIGNED = 3
WIN_MODE_1X1_PACKED = 4
WIN_MODE_FIRST_C3 = 5

PACK_CMD_VALID = 1 << 0
PACK_CMD_ZERO = 1 << 1
PACK_CMD_CONTIG_READ = 1 << 2
PACK_CMD_ALIGNED_READ = 1 << 3

ROW_CONSUMER_NONE = 0
ROW_CONSUMER_STORE = 1
ROW_CONSUMER_ADD_STORE = 2
ROW_CONSUMER_ADD_AFFINE_STORE = 3
ROW_CONSUMER_UPSAMPLE_OUT = 4

STORE_LAYOUT_NONE = 0
STORE_LAYOUT_COMPACT_C2 = 1
STORE_LAYOUT_COMPACT_C12 = 2
STORE_LAYOUT_COMPACT_C16 = 3
STORE_LAYOUT_COMPACT_C25 = 4
STORE_LAYOUT_COMPACT_C28 = 5
STORE_LAYOUT_C16_INTO_C19 = 6
STORE_LAYOUT_ALIGNED_TILE_COPY = 7
STORE_LAYOUT_NARROW_FIXED = 8
STORE_LAYOUT_COLD_RMW_FALLBACK = 15

STORE_LAYOUT_NAMES = {
    STORE_LAYOUT_NONE: "NONE",
    STORE_LAYOUT_COMPACT_C2: "COMPACT_C2",
    STORE_LAYOUT_COMPACT_C12: "COMPACT_C12",
    STORE_LAYOUT_COMPACT_C16: "COMPACT_C16",
    STORE_LAYOUT_COMPACT_C25: "COMPACT_C25",
    STORE_LAYOUT_COMPACT_C28: "COMPACT_C28",
    STORE_LAYOUT_C16_INTO_C19: "C16_INTO_C19",
    STORE_LAYOUT_ALIGNED_TILE_COPY: "ALIGNED_TILE_COPY",
    STORE_LAYOUT_NARROW_FIXED: "NARROW_FIXED",
    STORE_LAYOUT_COLD_RMW_FALLBACK: "COLD_RMW_FALLBACK",
}

EXEC_NOP = 0
EXEC_CONV = 1
EXEC_POOL = 2
EXEC_AFFINE = 3
EXEC_STORE = 4
EXEC_ADD_AFFINE = 5
EXEC_END = 255

FLAG_BIAS_EN = 1 << 0
FLAG_RELU_EN = 1 << 1
FLAG_REQUANT_BYPASS = 1 << 2
FLAG_CONCAT_MODE = 1 << 3
FLAG_ALIAS_ENABLE = 1 << 4
FLAG_POOL_SAME_SCALE = 1 << 5
FLAG_LAST_UOP_OF_STAGE = 1 << 6

TID_INVALID = 0xFF
LS_C1 = 0x80
LS_A = 0x81
LS_B = 0x82
LS_TMP = 0x83

BANK_FMEM0 = 0
BANK_FMEM1 = 1
BANK_FMEM2 = 2
BANK_BRAM_SCR0 = 0x80
BANK_BRAM_SCR1 = 0x81

FMBUF_POOL1_BASE = 0x3E0000
FMBUF_L20_BASE = 0x000000
FMBUF_L20_PHYS_C = 131
FMBUF_L20_C_OFFSET = 64
FMBUF_L30_BASE = 0x418000


@dataclass(frozen=True)
class TensorDesc:
    tensor_id: int
    name: str
    bank_id: int
    base_offset: int
    h: int
    w: int
    c: int
    phys_c: int
    c_offset: int


@dataclass(frozen=True)
class ConvSpec:
    param_id: int
    name: str
    input_scale: str
    output_scale: str
    out_c: int


@dataclass(frozen=True)
class AffineSpec:
    param_id: int
    name: str
    input_scale: str
    output_scale: str
    channels: int


@dataclass(frozen=True)
class AddSpec:
    param_id: int
    name: str
    src_a_scale: str
    src_b_scale: str
    dst_scale: str
    bypass: bool = True


@dataclass(frozen=True)
class PoolSpec:
    param_id: int
    name: str
    src_scale: str
    dst_scale: str
    same_scale: bool


@dataclass(frozen=True)
class Uop:
    opcode: int
    flags: int = 0
    src0: int = TID_INVALID
    src1: int = TID_INVALID
    dst: int = TID_INVALID
    param_id: int = 0
    act_type: int = ACT_NONE
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


@dataclass(frozen=True)
class WindowPackCmd:
    spatial_id: int
    src_c_begin: int
    dst_lane_begin: int
    byte_count: int
    flags: int = PACK_CMD_VALID | PACK_CMD_CONTIG_READ
    reserved0: int = 0
    reserved1: int = 0


@dataclass(frozen=True)
class WindowSchedDesc:
    mode: int
    kernel: int
    stride: int
    dilation: int
    in_c: int
    k_tiles: int
    cmd_base: int
    cmd_count: int
    kt_cmd_base: Tuple[int, ...]
    flags: int = 0


@dataclass(frozen=True)
class ConvExecDesc:
    param_id: int
    qparam_id: int
    window_sched_id: int
    row_consumer_id: int
    in_h: int
    in_w: int
    in_c: int
    out_c: int
    kernel: int
    stride: int
    dilation: int
    padding: int
    src_tensor: int
    dst_tensor: int
    dst_c_offset: int
    valid_c: int
    packed_weight_word_offset: int
    k_tiles: int
    weight_words: int
    flags: int = 0
    reserved: int = 0


@dataclass(frozen=True)
class RowConsumerDesc:
    mode: int
    add_other_tensor: int = TID_INVALID
    store_dst_tensor: int = TID_INVALID
    add_qparam_id: int = 0
    store_c_offset: int = 0
    valid_c: int = 0
    alias_tensor: int = TID_INVALID
    affine_param_id: int = 0
    affine_block_base: int = 0
    act_type: int = ACT_NONE
    reserved0: int = 0
    reserved1: int = 0


@dataclass(frozen=True)
class ExecPlanEntry:
    kind: int
    desc_id: int
    logical_uop_id: int
    flags: int = 0


@dataclass(frozen=True)
class PackedConvWeights:
    words: Tuple[bytes, ...]
    k_tiles: int
    active_oc: int


TENSORS: List[TensorDesc] = [
    TensorDesc(0, "T_INPUT", BANK_FMEM0, 0x000000, 512, 1024, 3, 3, 0),
    TensorDesc(1, "T_POOL1", BANK_FMEM0, FMBUF_POOL1_BASE, 256, 512, 3, 3, 0),
    TensorDesc(2, "T_B1_CAT", BANK_FMEM1, 0x180000, 256, 512, 19, 19, 0),
    TensorDesc(3, "T_B1_ACT", BANK_FMEM1, 0x180000, 256, 512, 19, 19, 0),
    TensorDesc(4, "T_L20_CAT", BANK_FMEM0, FMBUF_L20_BASE, 128, 256, 64, FMBUF_L20_PHYS_C, FMBUF_L20_C_OFFSET),
    TensorDesc(5, "T_L20_ACT", BANK_FMEM0, FMBUF_L20_BASE, 128, 256, 64, FMBUF_L20_PHYS_C, FMBUF_L20_C_OFFSET),
    TensorDesc(6, "T_L2B0_CAT", BANK_FMEM0, 0x000000, 128, 256, 64, 131, 0),
    TensorDesc(7, "T_L2B0_ACT", BANK_FMEM0, 0x000000, 128, 256, 64, 131, 0),
    TensorDesc(8, "T_POOL2", BANK_BRAM_SCR1, 0x000000, 128, 256, 3, 3, 0),
    TensorDesc(9, "T_B2_CAT", BANK_FMEM0, 0x000000, 128, 256, 131, 131, 0),
    TensorDesc(10, "T_B2_ACT", BANK_FMEM0, 0x000000, 128, 256, 131, 131, 0),
    TensorDesc(11, "T_L30_CAT", BANK_FMEM1, FMBUF_L30_BASE, 64, 128, 128, 128, 0),
    TensorDesc(12, "T_L30_ACT", BANK_FMEM1, FMBUF_L30_BASE, 64, 128, 128, 128, 0),
    TensorDesc(13, "T_L3B0_CAT", BANK_FMEM0, 0x000000, 64, 128, 128, 256, 128),
    TensorDesc(14, "T_L3B0_ACT", BANK_FMEM0, 0x000000, 64, 128, 128, 256, 128),
    TensorDesc(15, "T_B3_CAT", BANK_FMEM0, 0x000000, 64, 128, 256, 256, 0),
    TensorDesc(16, "T_B3_ACT", BANK_FMEM0, 0x000000, 64, 128, 256, 256, 0),
    TensorDesc(17, "T_OUT", BANK_FMEM0, 0x200000, 64, 128, 2, 2, 0),
    TensorDesc(18, "T_POOL_TMP", BANK_BRAM_SCR0, 0x000000, 256, 512, 3, 3, 0),
]


CONVS: List[ConvSpec] = [
    ConvSpec(0, "level1_conv", "quant", "b1_cat_ff", 16),
    ConvSpec(1, "level2_0_c1", "b1_bn", "level2_0_c1", 12),
    ConvSpec(2, "level2_0_d1", "level2_0_c1", "level2_0_cat_ff", 16),
    ConvSpec(3, "level2_0_d2", "level2_0_c1", "level2_0_cat_ff", 12),
    ConvSpec(4, "level2_0_d4", "level2_0_c1", "level2_0_cat_ff", 12),
    ConvSpec(5, "level2_0_d8", "level2_0_c1", "level2_0_cat_ff", 12),
    ConvSpec(6, "level2_0_d16", "level2_0_c1", "level2_0_cat_ff", 12),
    ConvSpec(7, "level2_blocks.0", "level2_0_bn", "level2_blocks.0", 12),
    ConvSpec(8, "level2_blocks.1", "level2_blocks.0", "level2_block_cat_ff.0", 16),
    ConvSpec(9, "level2_blocks.2", "level2_blocks.0", "level2_block_cat_ff.0", 12),
    ConvSpec(10, "level2_blocks.3", "level2_blocks.0", "level2_block_cat_ff.0", 12),
    ConvSpec(11, "level2_blocks.4", "level2_blocks.0", "level2_block_cat_ff.0", 12),
    ConvSpec(12, "level2_blocks.5", "level2_blocks.0", "level2_block_cat_ff.0", 12),
    ConvSpec(13, "level3_0_c1", "b2_bn", "level3_0_c1", 25),
    ConvSpec(14, "level3_0_d1", "level3_0_c1", "level3_0_cat_ff", 28),
    ConvSpec(15, "level3_0_d2", "level3_0_c1", "level3_0_cat_ff", 25),
    ConvSpec(16, "level3_0_d4", "level3_0_c1", "level3_0_cat_ff", 25),
    ConvSpec(17, "level3_0_d8", "level3_0_c1", "level3_0_cat_ff", 25),
    ConvSpec(18, "level3_0_d16", "level3_0_c1", "level3_0_cat_ff", 25),
    ConvSpec(19, "level3_blocks.0", "level3_0_bn", "level3_blocks.0", 25),
    ConvSpec(20, "level3_blocks.1", "level3_blocks.0", "level3_block_cat_ff.0", 28),
    ConvSpec(21, "level3_blocks.2", "level3_blocks.0", "level3_block_cat_ff.0", 25),
    ConvSpec(22, "level3_blocks.3", "level3_blocks.0", "level3_block_cat_ff.0", 25),
    ConvSpec(23, "level3_blocks.4", "level3_blocks.0", "level3_block_cat_ff.0", 25),
    ConvSpec(24, "level3_blocks.5", "level3_blocks.0", "level3_block_cat_ff.0", 25),
    ConvSpec(25, "classifier", "b3_bn", "classifier", 2),
]


AFFINES: List[AffineSpec] = [
    AffineSpec(0, "b1_bn", "b1_cat_ff", "b1_bn", 19),
    AffineSpec(1, "level2_0_bn", "level2_0_cat_ff", "level2_0_bn", 64),
    AffineSpec(2, "level2_blocks.6", "level2_block_res_ff.0", "level2_blocks.6", 64),
    AffineSpec(3, "b2_bn", "b2_cat_ff", "b2_bn", 131),
    AffineSpec(4, "level3_0_bn", "level3_0_cat_ff", "level3_0_bn", 128),
    AffineSpec(5, "level3_blocks.6", "level3_block_res_ff.0", "level3_blocks.6", 128),
    AffineSpec(6, "b3_bn", "b3_cat_ff", "b3_bn", 256),
]


ADDS: List[AddSpec] = [
    AddSpec(0, "L20_ADD_1", "level2_0_d2", "level2_0_d4", "level2_0_add2_ff"),
    AddSpec(1, "L20_ADD_2", "level2_0_add2_ff", "level2_0_d8", "level2_0_add3_ff"),
    AddSpec(2, "L20_ADD_3", "level2_0_add3_ff", "level2_0_d16", "level2_0_add4_ff"),
    AddSpec(3, "L2B0_ADD_1", "level2_blocks.2", "level2_blocks.3", "level2_block_add2_ff.0"),
    AddSpec(4, "L2B0_ADD_2", "level2_block_add2_ff.0", "level2_blocks.4", "level2_block_add3_ff.0"),
    AddSpec(5, "L2B0_ADD_3", "level2_block_add3_ff.0", "level2_blocks.5", "level2_block_add4_ff.0"),
    AddSpec(6, "L2B0_RES_ADD", "level2_block_cat_ff.0", "level2_0_bn", "level2_block_res_ff.0"),
    AddSpec(7, "L30_ADD_1", "level3_0_d2", "level3_0_d4", "level3_0_add2_ff"),
    AddSpec(8, "L30_ADD_2", "level3_0_add2_ff", "level3_0_d8", "level3_0_add3_ff"),
    AddSpec(9, "L30_ADD_3", "level3_0_add3_ff", "level3_0_d16", "level3_0_add4_ff"),
    AddSpec(10, "L3B0_ADD_1", "level3_blocks.2", "level3_blocks.3", "level3_block_add2_ff.0"),
    AddSpec(11, "L3B0_ADD_2", "level3_block_add2_ff.0", "level3_blocks.4", "level3_block_add3_ff.0"),
    AddSpec(12, "L3B0_ADD_3", "level3_block_add3_ff.0", "level3_blocks.5", "level3_block_add4_ff.0"),
    AddSpec(13, "L3B0_RES_ADD", "level3_block_cat_ff.0", "level3_0_bn", "level3_block_res_ff.0"),
]


POOLS: List[PoolSpec] = [
    PoolSpec(0, "POOL_B1", "quant", "b1_cat_ff", False),
    PoolSpec(1, "POOL_B2_TMP", "quant", "quant", True),
    PoolSpec(2, "POOL_B2_OUT", "quant", "b2_cat_ff", False),
]

# P7 compiler-side names.  Keep the legacy constants above for compatibility
# with the existing P6 blob path and CSim comparisons.
TENSOR_PLAN = TENSORS
CONV_PLAN = CONVS
AFFINE_PLAN = AFFINES
ADD_PLAN = ADDS
POOL_PLAN = POOLS


def safe_name(name: str) -> str:
    return name.replace(".", "_")


def round_away_from_zero(x: float) -> int:
    if x >= 0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def align_up(value: int, align: int = SECTION_ALIGN) -> int:
    return (value + align - 1) // align * align


def pad_to(buf: bytearray, align: int = SECTION_ALIGN) -> None:
    buf.extend(b"\x00" * (align_up(len(buf), align) - len(buf)))


def choose_multiplier(real_multiplier: float, max_shift: int = 31) -> Tuple[int, int]:
    if not math.isfinite(real_multiplier):
        raise ValueError(f"Non-finite multiplier: {real_multiplier}")
    if real_multiplier == 0.0:
        return 0, 0

    sign = -1 if real_multiplier < 0 else 1
    value = abs(real_multiplier)
    for shift in range(max_shift, -1, -1):
        mult = round_away_from_zero(value * (1 << shift))
        if 0 < mult <= 0x7FFFFFFF:
            return sign * mult, shift

    mult = min(round_away_from_zero(value), 0x7FFFFFFF)
    return sign * mult, 0


def choose_affine_fixed_point(a: float, b: float, max_shift: int = 31) -> Tuple[int, int, int]:
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError(f"Non-finite affine fixed-point terms: a={a}, b={b}")

    for shift in range(max_shift, -1, -1):
        mult = round_away_from_zero(a * (1 << shift))
        bias = round_away_from_zero(b * (1 << shift))
        if -0x80000000 <= mult <= 0x7FFFFFFF and -0x80000000 <= bias <= 0x7FFFFFFF:
            return mult, bias, shift

    raise OverflowError(f"Cannot fit affine fixed-point terms into int32: a={a}, b={b}")


def pack_tensor_desc(desc: TensorDesc) -> bytes:
    return struct.pack(
        "<BBHIHHHH",
        desc.bank_id,
        1,
        desc.phys_c,
        desc.base_offset,
        desc.h,
        desc.w,
        desc.c,
        desc.c_offset,
    )


def pack_scale_desc(scale: float) -> bytes:
    mult, shift = choose_multiplier(scale)
    return struct.pack("<iB3x", mult, shift)


def pack_param_desc(offset: int, second: int = 0, third: int = 0) -> bytes:
    return struct.pack("<4I", offset, second, third, 0)


def pack_uop(uop: Uop) -> bytes:
    return struct.pack(
        "<8B4H4B4HI",
        uop.opcode,
        uop.flags,
        uop.src0,
        uop.src1,
        uop.dst,
        uop.param_id,
        uop.act_type,
        0,
        uop.in_h,
        uop.in_w,
        uop.in_c,
        uop.out_c,
        uop.kernel,
        uop.stride,
        uop.dilation,
        uop.padding,
        uop.c_offset,
        uop.valid_c,
        uop.qparam_id,
        0,
        0,
    )


def pack_window_pack_cmd(cmd: WindowPackCmd) -> bytes:
    return struct.pack(
        "<6BH",
        cmd.spatial_id,
        cmd.src_c_begin,
        cmd.dst_lane_begin,
        cmd.byte_count,
        cmd.flags,
        cmd.reserved0,
        cmd.reserved1,
    )


def pack_window_sched_desc(desc: WindowSchedDesc) -> bytes:
    kt_bases = list(desc.kt_cmd_base)
    if len(kt_bases) > MAX_K_TILE_COUNT + 1:
        raise ValueError(f"window schedule has too many kt offsets: {len(kt_bases)}")
    kt_bases += [kt_bases[-1] if kt_bases else 0] * (MAX_K_TILE_COUNT + 1 - len(kt_bases))
    return (
        struct.pack(
            "<4B4H",
            desc.mode,
            desc.kernel,
            desc.stride,
            desc.dilation,
            desc.in_c,
            desc.k_tiles,
            desc.cmd_base,
            desc.cmd_count,
        )
        + struct.pack("<41H", *kt_bases)
        + struct.pack("<H", desc.flags)
    )


def pack_conv_exec_desc(desc: ConvExecDesc) -> bytes:
    return struct.pack(
        "<4B4H4B2B2HI4H2x",
        desc.param_id,
        desc.qparam_id,
        desc.window_sched_id,
        desc.row_consumer_id,
        desc.in_h,
        desc.in_w,
        desc.in_c,
        desc.out_c,
        desc.kernel,
        desc.stride,
        desc.dilation,
        desc.padding,
        desc.src_tensor,
        desc.dst_tensor,
        desc.dst_c_offset,
        desc.valid_c,
        desc.packed_weight_word_offset,
        desc.k_tiles,
        desc.weight_words,
        desc.flags,
        desc.reserved,
    )


def pack_row_consumer_desc(desc: RowConsumerDesc) -> bytes:
    return struct.pack(
        "<4B2H4B2H",
        desc.mode,
        desc.add_other_tensor,
        desc.store_dst_tensor,
        desc.add_qparam_id,
        desc.store_c_offset,
        desc.valid_c,
        desc.alias_tensor,
        desc.affine_param_id,
        desc.affine_block_base,
        desc.act_type,
        desc.reserved0,
        desc.reserved1,
    )


def pack_exec_plan_entry(entry: ExecPlanEntry) -> bytes:
    return struct.pack("<4B", entry.kind, entry.desc_id, entry.logical_uop_id, entry.flags)


def pack_conv_qparam(bias: Sequence[int], mult: Sequence[int], shift: Sequence[int]) -> bytes:
    b = list(bias)[:32] + [0] * (32 - len(bias))
    m = list(mult)[:32] + [0] * (32 - len(mult))
    s = list(shift)[:32] + [0] * (32 - len(shift))
    return struct.pack("<32i32i32B32B", *b, *m, *s, *([0] * 32))


def pack_affine_qparam(mul: Sequence[int], bias: Sequence[int], shift: Sequence[int]) -> bytes:
    m = list(mul)[:32] + [0] * (32 - len(mul))
    b = list(bias)[:32] + [0] * (32 - len(bias))
    s = list(shift)[:32] + [0] * (32 - len(shift))
    return struct.pack("<32i32i32B32B", *m, *b, *s, *([0] * 32))


def pack_add_qparam(
    mult: int,
    shift: int,
    act_type: int,
    requant_bypass: bool,
    src_a_id: int,
    src_b_id: int,
    dst_id: int,
) -> bytes:
    return struct.pack(
        "<i4B4I",
        mult,
        shift,
        act_type,
        1 if requant_bypass else 0,
        0,
        src_a_id,
        src_b_id,
        dst_id,
        0,
    )


def pack_pool_qparam(
    kernel: int,
    stride: int,
    same_scale: bool,
    act_type: int,
    src_scale_id: int,
    dst_scale_id: int,
    mult: int,
    shift: int,
) -> bytes:
    return struct.pack(
        "<4BIIiB11B",
        kernel,
        stride,
        1 if same_scale else 0,
        act_type,
        src_scale_id,
        dst_scale_id,
        mult,
        shift,
        *([0] * 11),
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scalar_qparam_from_record(record: dict, name: str) -> Tuple[float, int]:
    scale = record.get("scale")
    zp = record.get("zero_point")
    if not scale or not zp:
        raise KeyError(f"Missing activation scale/zero_point for {name}")
    if len(scale) != 1 or len(zp) != 1:
        raise ValueError(f"Only per-tensor activation qparams are supported for {name}")
    return float(scale[0]), int(zp[0])


def collect_activation_qparams(artifact_dir: Path) -> Dict[str, Tuple[float, int, str]]:
    activation_records = load_json(artifact_dir / "activation_qparams.json")
    golden_dir = artifact_dir / "golden_sample"
    result: Dict[str, Tuple[float, int, str]] = {}

    needed = set()
    for spec in CONVS:
        needed.add(spec.input_scale)
        needed.add(spec.output_scale)
        needed.add(spec.name)
    for spec in AFFINES:
        needed.add(spec.input_scale)
        needed.add(spec.output_scale)
        needed.add(spec.name)
    for spec in ADDS:
        needed.add(spec.src_a_scale)
        needed.add(spec.src_b_scale)
        needed.add(spec.dst_scale)
    for spec in POOLS:
        needed.add(spec.src_scale)
        needed.add(spec.dst_scale)

    for name in sorted(needed):
        qparam_path = golden_dir / safe_name(name) / "qparams.json"
        if qparam_path.exists():
            scale, zp = scalar_qparam_from_record(load_json(qparam_path), name)
            result[name] = (scale, zp, "golden_sample")
            continue

        if name in activation_records:
            scale, zp = scalar_qparam_from_record(activation_records[name], name)
            result[name] = (scale, zp, "activation_qparams")
            continue

        raise KeyError(f"No activation qparams found for {name}")

    return result


def build_scale_table(qparams: Dict[str, Tuple[float, int, str]]) -> Tuple[List[str], Dict[str, int]]:
    ordered: List[str] = []

    def add(name: str) -> None:
        if name not in ordered:
            ordered.append(name)

    add("quant")
    for spec in CONVS:
        add(spec.input_scale)
        add(spec.output_scale)
        add(spec.name)
    for spec in AFFINES:
        add(spec.input_scale)
        add(spec.output_scale)
    for spec in ADDS:
        add(spec.src_a_scale)
        add(spec.src_b_scale)
        add(spec.dst_scale)
    for spec in POOLS:
        add(spec.src_scale)
        add(spec.dst_scale)

    if len(ordered) > 128:
        raise ValueError(f"scale_desc_count exceeds HLS max: {len(ordered)}")
    return ordered, {name: idx for idx, name in enumerate(ordered)}


def require_int8_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.min() < -128 or arr.max() > 127:
        raise ValueError(f"{path} contains values outside int8 range")
    return arr.astype(np.int8, copy=False)


def linear_k_to_spatial_c(k: int, in_c: int, kernel: int) -> Tuple[int, int, int, int]:
    if in_c <= 0 or kernel not in (1, 3):
        raise ValueError(f"unsupported K layout: in_c={in_c}, kernel={kernel}")
    spatial = k // in_c
    cin = k % in_c
    kh = spatial // kernel
    kw = spatial % kernel
    return spatial, kh, kw, cin


TENSOR_DESC_BY_ID: Dict[int, TensorDesc] = {desc.tensor_id: desc for desc in TENSORS}


def tensor_desc_supports_aligned_1x1_read(desc: TensorDesc) -> bool:
    return (
        desc.phys_c >= TK
        and desc.phys_c % TK == 0
        and (desc.base_offset + desc.c_offset) % TK == 0
    )


def tensor_supports_aligned_1x1_read(tensor_id: int) -> bool:
    desc = TENSOR_DESC_BY_ID.get(tensor_id)
    return False if desc is None else tensor_desc_supports_aligned_1x1_read(desc)


def tensor_is_scratch_id(tensor_id: int) -> bool:
    return tensor_id in (LS_C1, LS_A, LS_B, LS_TMP)


def scratch_index(tensor_id: int) -> int:
    return {LS_C1: 0, LS_A: 1, LS_B: 2, LS_TMP: 3}[tensor_id]


def conv_out_dim(size: int, stride: int) -> int:
    eff_stride = 1 if stride == 0 else stride
    return (size + eff_stride - 1) // eff_stride


def scratch_desc_for_conv_store(tensor_id: int, producer: Uop, valid_c: int) -> TensorDesc:
    idx = scratch_index(tensor_id)
    pid = producer.param_id
    if 1 <= pid <= 6:
        if idx == 0:
            phys_c, c_offset = valid_c, 0
        else:
            phys_c, c_offset = FMBUF_L20_PHYS_C, (idx - 1) * 16
    elif 7 <= pid <= 12:
        phys_c, c_offset = valid_c, 0
    elif 13 <= pid <= 18:
        phys_c, c_offset = valid_c, 0
    elif 19 <= pid <= 24:
        phys_c, c_offset = 256, idx * 32
    else:
        phys_c, c_offset = valid_c, 0
    return TensorDesc(
        tensor_id=tensor_id,
        name=f"scratch_{tensor_id:02x}",
        bank_id=BANK_FMEM0,
        base_offset=0,
        h=conv_out_dim(producer.in_h, producer.stride),
        w=conv_out_dim(producer.in_w, producer.stride),
        c=valid_c,
        phys_c=phys_c,
        c_offset=c_offset,
    )


def resolve_compile_time_store_desc(dst_tensor: int, producer: Uop, valid_c: int) -> TensorDesc:
    if tensor_is_scratch_id(dst_tensor):
        return scratch_desc_for_conv_store(dst_tensor, producer, valid_c)
    desc = TENSOR_DESC_BY_ID.get(dst_tensor)
    if desc is None:
        raise ValueError(f"store layout references unknown tensor id {dst_tensor}")
    return desc


def store_layout_for(
    *,
    dst_tensor: int,
    producer: Uop,
    c_offset: int,
    valid_c: int,
    out_w: int,
) -> int:
    if dst_tensor == TID_INVALID:
        return STORE_LAYOUT_NONE
    desc = resolve_compile_time_store_desc(dst_tensor, producer, valid_c)
    if valid_c <= 0 or valid_c > TM:
        raise ValueError(
            f"store layout only supports row-buffer slices with 1..{TM} channels, "
            f"got tensor={dst_tensor} valid_c={valid_c}"
        )

    compact_full_row = c_offset == 0 and desc.c == valid_c and desc.phys_c == desc.c
    if compact_full_row:
        if valid_c == 2 and out_w % 16 == 0:
            return STORE_LAYOUT_COMPACT_C2
        if valid_c == 12 and out_w % 8 == 0:
            return STORE_LAYOUT_COMPACT_C12
        if valid_c == 16 and out_w % 2 == 0:
            return STORE_LAYOUT_COMPACT_C16
        if valid_c == 25 and out_w % 32 == 0:
            return STORE_LAYOUT_COMPACT_C25
        if valid_c == 28 and out_w % 8 == 0:
            return STORE_LAYOUT_COMPACT_C28

    if c_offset == 0 and valid_c == 16 and desc.c == 19 and desc.phys_c == 19 and out_w % 32 == 0:
        return STORE_LAYOUT_C16_INTO_C19

    start_c = desc.c_offset + c_offset
    if valid_c == TM and desc.phys_c % TM == 0 and start_c % TM == 0:
        return STORE_LAYOUT_ALIGNED_TILE_COPY
    if valid_c <= TM:
        return STORE_LAYOUT_NARROW_FIXED
    return STORE_LAYOUT_COLD_RMW_FALLBACK


def window_mode_for(in_c: int, kernel: int, aligned_1x1_source: bool = True) -> int:
    if kernel == 1:
        return WIN_MODE_1X1_ALIGNED if in_c % TK == 0 and aligned_1x1_source else WIN_MODE_1X1_PACKED
    if kernel != 3:
        raise ValueError(f"unsupported kernel size: {kernel}")
    if in_c == 3:
        return WIN_MODE_FIRST_C3
    if in_c <= 32:
        return WIN_MODE_SMALLC_3X3_STAGED
    return WIN_MODE_LARGEC_3X3_SEGMENT


def make_window_pack_schedule(
    in_c: int,
    kernel: int,
    tk: int = TK,
    mode: Optional[int] = None,
) -> Tuple[WindowSchedDesc, List[WindowPackCmd]]:
    k_total = kernel * kernel * in_c
    k_tiles = math.ceil(k_total / tk)
    if k_tiles > MAX_K_TILE_COUNT:
        raise ValueError(f"k_tiles={k_tiles} exceeds MAX_K_TILE_COUNT={MAX_K_TILE_COUNT}")

    commands: List[WindowPackCmd] = []
    kt_cmd_base: List[int] = []
    for kt in range(k_tiles):
        kt_cmd_base.append(len(commands))
        k = kt * tk
        kt_end = min(k + tk, k_total)
        while k < kt_end:
            spatial, _kh, _kw, cin = linear_k_to_spatial_c(k, in_c, kernel)
            dst_lane = k - kt * tk
            count = min(kt_end - k, in_c - cin, tk - dst_lane)
            if count <= 0:
                raise AssertionError("zero-length window pack command")
            commands.append(
                WindowPackCmd(
                    spatial_id=spatial,
                    src_c_begin=cin,
                    dst_lane_begin=dst_lane,
                    byte_count=count,
                    flags=PACK_CMD_VALID | PACK_CMD_CONTIG_READ,
                )
            )
            k += count
        if len(commands) - kt_cmd_base[-1] > MAX_PACK_CMDS_PER_KT:
            raise ValueError(
                f"in_c={in_c} kernel={kernel} kt={kt} uses "
                f"{len(commands) - kt_cmd_base[-1]} commands, max={MAX_PACK_CMDS_PER_KT}"
            )
    kt_cmd_base.append(len(commands))

    desc = WindowSchedDesc(
        mode=window_mode_for(in_c, kernel) if mode is None else mode,
        kernel=kernel,
        stride=0,
        dilation=0,
        in_c=in_c,
        k_tiles=k_tiles,
        cmd_base=0,
        cmd_count=len(commands),
        kt_cmd_base=tuple(kt_cmd_base),
    )
    return desc, commands


def pack_single_conv_weights(weight: np.ndarray, out_c: int, in_c: int, kernel: int) -> PackedConvWeights:
    if weight.shape[:2] != (out_c, in_c) or weight.shape[2:] != (kernel, kernel):
        raise ValueError(
            f"weight shape {weight.shape} does not match out_c={out_c}, in_c={in_c}, kernel={kernel}"
        )
    k_total = kernel * kernel * in_c
    k_tiles = math.ceil(k_total / TK)
    words: List[bytes] = []
    for oc in range(out_c):
        for kt in range(k_tiles):
            word = bytearray(TK)
            for lane in range(TK):
                k = kt * TK + lane
                if k < k_total:
                    _spatial, kh, kw, cin = linear_k_to_spatial_c(k, in_c, kernel)
                    word[lane] = int(weight[oc, cin, kh, kw]) & 0xFF
            words.append(bytes(word))
    return PackedConvWeights(tuple(words), k_tiles=k_tiles, active_oc=out_c)


def build_packed_weight_section(artifact_dir: Path, warnings: List[str]) -> Tuple[bytearray, Dict[int, int], List[dict]]:
    weight_blob = bytearray()
    word_offsets: Dict[int, int] = {}
    report: List[dict] = []

    for spec in CONV_PLAN:
        layer_dir = artifact_dir / "layers" / safe_name(spec.name)
        weight = require_int8_array(layer_dir / "weight_int8.npy")
        out_c, in_c, kh, kw = weight.shape
        if out_c != spec.out_c or kh != kw or kh not in (1, 3):
            raise ValueError(f"{spec.name}: unsupported weight shape {weight.shape}, spec out_c={spec.out_c}")

        zp_path = layer_dir / "weight_zero_points.npy"
        if zp_path.exists():
            zps = np.load(zp_path).reshape(-1)
            non_zero = sorted({int(x) for x in zps.tolist() if int(x) != 0})
            if non_zero:
                warnings.append(
                    f"{spec.name}: exported weight_zero_points has non-zero entries {non_zero}; "
                    "weight_int8.npy is still treated as symmetric signed INT8."
                )

        pad_to(weight_blob)
        word_offsets[spec.param_id] = len(weight_blob) // TK
        packed = pack_single_conv_weights(weight, out_c=out_c, in_c=in_c, kernel=kh)
        for word in packed.words:
            if len(word) != TK:
                raise AssertionError("packed weight word is not 32 bytes")
            weight_blob.extend(word)
        report.append(
            {
                "param_id": spec.param_id,
                "name": spec.name,
                "shape_oihw": [int(x) for x in weight.shape],
                "packed_word_offset": word_offsets[spec.param_id],
                "packed_words": len(packed.words),
                "k_tiles": packed.k_tiles,
                "active_oc": packed.active_oc,
                "layout": "active_oc_major_then_kt; invalid tm lanes are implicit zero in HLS",
            }
        )

    pad_to(weight_blob)
    if len(weight_blob) > WBUF_BYTES:
        raise ValueError(f"packed weights use {len(weight_blob)} bytes, WBUF_BYTES={WBUF_BYTES}")
    return weight_blob, word_offsets, report


def build_weight_section(artifact_dir: Path, warnings: List[str]) -> Tuple[bytearray, Dict[int, int]]:
    weight_blob = bytearray()
    local_offsets: Dict[int, int] = {}

    for spec in CONVS:
        layer_dir = artifact_dir / "layers" / safe_name(spec.name)
        weight = require_int8_array(layer_dir / "weight_int8.npy")
        if weight.shape[0] != spec.out_c:
            raise ValueError(f"{spec.name}: expected out_c={spec.out_c}, got weight shape {weight.shape}")

        zp_path = layer_dir / "weight_zero_points.npy"
        if zp_path.exists():
            zps = np.load(zp_path).reshape(-1)
            non_zero = sorted({int(x) for x in zps.tolist() if int(x) != 0})
            if non_zero:
                warnings.append(
                    f"{spec.name}: exported weight_zero_points has non-zero entries {non_zero}; "
                    "weight_int8.npy is still treated as symmetric signed INT8."
                )

        pad_to(weight_blob)
        local_offsets[spec.param_id] = len(weight_blob)
        weight_blob.extend(np.ascontiguousarray(weight).tobytes(order="C"))

    pad_to(weight_blob)
    return weight_blob, local_offsets


def get_state_tensor(state: dict, key: str) -> np.ndarray:
    if key not in state:
        raise KeyError(f"Missing state_dict key: {key}")
    return state[key].detach().cpu().numpy()


def build_conv_qparams(
    artifact_dir: Path,
    state: dict,
    qparams: Dict[str, Tuple[float, int, str]],
) -> Tuple[bytearray, Dict[int, int], List[dict]]:
    blob = bytearray()
    local_offsets: Dict[int, int] = {}
    report: List[dict] = []

    for spec in CONVS:
        layer_dir = artifact_dir / "layers" / safe_name(spec.name)
        weight_scales = np.load(layer_dir / "weight_scales.npy").astype(np.float64).reshape(-1)
        if len(weight_scales) != spec.out_c:
            raise ValueError(f"{spec.name}: expected {spec.out_c} weight scales, got {len(weight_scales)}")

        input_scale, input_zp, _ = qparams[spec.input_scale]
        output_scale, output_zp, _ = qparams[spec.output_scale]
        if input_zp != 0:
            raise ValueError(f"{spec.name}: conv input zero_point is {input_zp}, hardware conv expects 0")

        bias_key = f"{spec.name}.bias"
        if bias_key in state:
            bias_fp = get_state_tensor(state, bias_key).astype(np.float64).reshape(-1)
            if len(bias_fp) != spec.out_c:
                raise ValueError(f"{spec.name}: bias length mismatch")
        else:
            bias_fp = np.zeros(spec.out_c, dtype=np.float64)

        bias_i32: List[int] = []
        mult_i32: List[int] = []
        shift_u8: List[int] = []
        for ch in range(spec.out_c):
            scale_acc = input_scale * float(weight_scales[ch])
            ratio = scale_acc / output_scale
            mult, shift = choose_multiplier(ratio)

            # Fold output zero-point into the pre-scale bias term when needed.
            zp_bias = 0.0 if output_zp == 0 else (output_zp / ratio)
            bias_i32.append(round_away_from_zero(float(bias_fp[ch]) / scale_acc + zp_bias))
            mult_i32.append(mult)
            shift_u8.append(shift)

        local_offsets[spec.param_id] = len(blob)
        blob.extend(pack_conv_qparam(bias_i32, mult_i32, shift_u8))
        report.append(
            {
                "param_id": spec.param_id,
                "name": spec.name,
                "input_scale": spec.input_scale,
                "output_scale": spec.output_scale,
                "output_zero_point": output_zp,
                "channels": spec.out_c,
            }
        )

    pad_to(blob)
    return blob, local_offsets, report


def build_affine_qparams(
    state: dict,
    qparams: Dict[str, Tuple[float, int, str]],
) -> Tuple[bytearray, Dict[int, int], List[dict]]:
    blob = bytearray()
    local_offsets: Dict[int, int] = {}
    report: List[dict] = []
    eps = 1.0e-3

    for spec in AFFINES:
        gamma = get_state_tensor(state, f"{spec.name}.weight").astype(np.float64).reshape(-1)
        beta = get_state_tensor(state, f"{spec.name}.bias").astype(np.float64).reshape(-1)
        mean = get_state_tensor(state, f"{spec.name}.running_mean").astype(np.float64).reshape(-1)
        var = get_state_tensor(state, f"{spec.name}.running_var").astype(np.float64).reshape(-1)
        if len(gamma) != spec.channels:
            raise ValueError(f"{spec.name}: expected {spec.channels} affine channels, got {len(gamma)}")

        input_scale, input_zp, _ = qparams[spec.input_scale]
        output_scale, output_zp, _ = qparams[spec.output_scale]
        alpha = gamma / np.sqrt(var + eps)
        beta_term = beta - mean * alpha

        local_offsets[spec.param_id] = len(blob)
        for base in range(0, spec.channels, 32):
            mul_i32: List[int] = []
            bias_i32: List[int] = []
            shift_u8: List[int] = []
            for ch in range(base, min(base + 32, spec.channels)):
                a = input_scale * float(alpha[ch]) / output_scale
                b = float(beta_term[ch]) / output_scale + output_zp - a * input_zp
                mult, bias, shift = choose_affine_fixed_point(a, b)
                mul_i32.append(mult)
                bias_i32.append(bias)
                shift_u8.append(shift)
            blob.extend(pack_affine_qparam(mul_i32, bias_i32, shift_u8))

        report.append(
            {
                "param_id": spec.param_id,
                "name": spec.name,
                "input_scale": spec.input_scale,
                "input_zero_point": input_zp,
                "output_scale": spec.output_scale,
                "output_zero_point": output_zp,
                "channels": spec.channels,
                "blocks_32ch": math.ceil(spec.channels / 32),
            }
        )

    pad_to(blob)
    return blob, local_offsets, report


def almost_same(a: float, b: float, rel_tol: float = 1e-5, abs_tol: float = 1e-8) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def build_add_qparams(
    qparams: Dict[str, Tuple[float, int, str]],
    scale_ids: Dict[str, int],
    warnings: List[str],
) -> Tuple[bytearray, Dict[int, int], List[dict]]:
    blob = bytearray()
    local_offsets: Dict[int, int] = {}
    report: List[dict] = []

    for spec in ADDS:
        a_scale, a_zp, _ = qparams[spec.src_a_scale]
        b_scale, b_zp, _ = qparams[spec.src_b_scale]
        d_scale, d_zp, _ = qparams[spec.dst_scale]
        if not (a_zp == b_zp == d_zp == 0):
            warnings.append(
                f"{spec.name}: ADD qparams include non-zero zp "
                f"({spec.src_a_scale}:{a_zp}, {spec.src_b_scale}:{b_zp}, {spec.dst_scale}:{d_zp})."
            )
        if spec.bypass and not (almost_same(a_scale, b_scale) and almost_same(a_scale, d_scale)):
            warnings.append(
                f"{spec.name}: requant_bypass=1 but scales differ "
                f"({spec.src_a_scale}={a_scale:.8g}, {spec.src_b_scale}={b_scale:.8g}, "
                f"{spec.dst_scale}={d_scale:.8g}). This reflects current QAT artifacts, "
                "not the frozen zero-requant ADD assumption."
            )

        local_offsets[spec.param_id] = len(blob)
        blob.extend(
            pack_add_qparam(
                mult=1,
                shift=0,
                act_type=ACT_NONE,
                requant_bypass=spec.bypass,
                src_a_id=scale_ids[spec.src_a_scale],
                src_b_id=scale_ids[spec.src_b_scale],
                dst_id=scale_ids[spec.dst_scale],
            )
        )
        report.append(
            {
                "param_id": spec.param_id,
                "name": spec.name,
                "src_a_scale": spec.src_a_scale,
                "src_b_scale": spec.src_b_scale,
                "dst_scale": spec.dst_scale,
                "requant_bypass": spec.bypass,
            }
        )

    pad_to(blob)
    return blob, local_offsets, report


def build_pool_qparams(
    qparams: Dict[str, Tuple[float, int, str]],
    scale_ids: Dict[str, int],
) -> Tuple[bytearray, Dict[int, int], List[dict]]:
    blob = bytearray()
    local_offsets: Dict[int, int] = {}
    report: List[dict] = []

    for spec in POOLS:
        src_scale, src_zp, _ = qparams[spec.src_scale]
        dst_scale, dst_zp, _ = qparams[spec.dst_scale]
        if src_zp != 0 or dst_zp != 0:
            raise ValueError(f"{spec.name}: pool zero-points must be 0")
        if spec.same_scale:
            mult, shift = 1, 0
        else:
            mult, shift = choose_multiplier(src_scale / dst_scale)

        local_offsets[spec.param_id] = len(blob)
        blob.extend(
            pack_pool_qparam(
                kernel=3,
                stride=2,
                same_scale=spec.same_scale,
                act_type=ACT_NONE,
                src_scale_id=scale_ids[spec.src_scale],
                dst_scale_id=scale_ids[spec.dst_scale],
                mult=mult,
                shift=shift,
            )
        )
        report.append(
            {
                "param_id": spec.param_id,
                "name": spec.name,
                "src_scale": spec.src_scale,
                "dst_scale": spec.dst_scale,
                "same_scale": spec.same_scale,
                "mult": mult,
                "shift": shift,
            }
        )

    pad_to(blob)
    return blob, local_offsets, report


def stage_last(uop: Uop) -> Uop:
    return Uop(**{**uop.__dict__, "flags": uop.flags | FLAG_LAST_UOP_OF_STAGE})


def build_uops() -> List[Uop]:
    cflag = FLAG_CONCAT_MODE
    aflag = FLAG_ALIAS_ENABLE
    addflag = FLAG_REQUANT_BYPASS
    pool_same = FLAG_POOL_SAME_SCALE

    uops = [
        Uop(UOP_LOAD_FM, dst=0, in_h=512, in_w=1024, in_c=3, out_c=3, valid_c=3),
        Uop(UOP_POOL, src0=0, dst=1, param_id=0, in_h=512, in_w=1024, in_c=3, out_c=3, kernel=3, stride=2, padding=1, valid_c=3),
        Uop(UOP_CONV, flags=FLAG_BIAS_EN | FLAG_RELU_EN | cflag, src0=0, dst=2, param_id=0, act_type=ACT_RELU, in_h=512, in_w=1024, in_c=3, out_c=16, kernel=3, stride=2, dilation=1, padding=1, valid_c=16),
        Uop(UOP_STORE, flags=cflag, src0=1, dst=2, in_h=256, in_w=512, in_c=3, out_c=19, c_offset=16, valid_c=3),
        stage_last(Uop(UOP_AFFINE, flags=aflag, src0=2, dst=3, param_id=0, act_type=ACT_RELU, in_h=256, in_w=512, in_c=19, out_c=19, valid_c=19)),
        Uop(UOP_POOL, flags=pool_same, src0=0, dst=18, param_id=1, in_h=512, in_w=1024, in_c=3, out_c=3, kernel=3, stride=2, padding=1, valid_c=3),
        Uop(UOP_POOL, src0=18, dst=8, param_id=2, in_h=256, in_w=512, in_c=3, out_c=3, kernel=3, stride=2, padding=1, valid_c=3),
        Uop(UOP_CONV, src0=3, dst=LS_C1, param_id=1, in_h=256, in_w=512, in_c=19, out_c=12, kernel=3, stride=2, dilation=1, padding=1, valid_c=12),
        Uop(UOP_CONV, flags=cflag, src0=LS_C1, dst=4, param_id=2, in_h=128, in_w=256, in_c=12, out_c=16, kernel=3, stride=1, dilation=1, padding=1, c_offset=0, valid_c=16),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_A, param_id=3, in_h=128, in_w=256, in_c=12, out_c=12, kernel=3, stride=1, dilation=2, padding=2, valid_c=12),
        Uop(UOP_STORE, flags=cflag, src0=LS_A, dst=4, in_h=128, in_w=256, in_c=12, out_c=64, c_offset=16, valid_c=12),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=4, in_h=128, in_w=256, in_c=12, out_c=12, kernel=3, stride=1, dilation=4, padding=4, valid_c=12),
        Uop(UOP_ADD, flags=addflag, src0=LS_A, src1=LS_TMP, dst=LS_B, param_id=0, in_h=128, in_w=256, in_c=12, out_c=12, valid_c=12),
        Uop(UOP_STORE, flags=cflag, src0=LS_B, dst=4, in_h=128, in_w=256, in_c=12, out_c=64, c_offset=28, valid_c=12),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=5, in_h=128, in_w=256, in_c=12, out_c=12, kernel=3, stride=1, dilation=8, padding=8, valid_c=12),
        Uop(UOP_ADD, flags=addflag, src0=LS_B, src1=LS_TMP, dst=LS_A, param_id=1, in_h=128, in_w=256, in_c=12, out_c=12, valid_c=12),
        Uop(UOP_STORE, flags=cflag, src0=LS_A, dst=4, in_h=128, in_w=256, in_c=12, out_c=64, c_offset=40, valid_c=12),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=6, in_h=128, in_w=256, in_c=12, out_c=12, kernel=3, stride=1, dilation=16, padding=16, valid_c=12),
        Uop(UOP_ADD, flags=addflag, src0=LS_A, src1=LS_TMP, dst=LS_B, param_id=2, in_h=128, in_w=256, in_c=12, out_c=12, valid_c=12),
        Uop(UOP_STORE, flags=cflag, src0=LS_B, dst=4, in_h=128, in_w=256, in_c=12, out_c=64, c_offset=52, valid_c=12),
        stage_last(Uop(UOP_AFFINE, flags=aflag, src0=4, dst=5, param_id=1, act_type=ACT_RELU, in_h=128, in_w=256, in_c=64, out_c=64, valid_c=64)),
        Uop(UOP_CONV, src0=5, dst=LS_C1, param_id=7, in_h=128, in_w=256, in_c=64, out_c=12, kernel=1, stride=1, dilation=1, padding=0, valid_c=12),
        Uop(UOP_CONV, flags=cflag, src0=LS_C1, dst=6, param_id=8, in_h=128, in_w=256, in_c=12, out_c=16, kernel=3, stride=1, dilation=1, padding=1, c_offset=0, valid_c=16),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_A, param_id=9, in_h=128, in_w=256, in_c=12, out_c=12, kernel=3, stride=1, dilation=2, padding=2, valid_c=12),
        Uop(UOP_STORE, flags=cflag, src0=LS_A, dst=6, in_h=128, in_w=256, in_c=12, out_c=64, c_offset=16, valid_c=12),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=10, in_h=128, in_w=256, in_c=12, out_c=12, kernel=3, stride=1, dilation=4, padding=4, valid_c=12),
        Uop(UOP_ADD, flags=addflag, src0=LS_A, src1=LS_TMP, dst=LS_B, param_id=3, in_h=128, in_w=256, in_c=12, out_c=12, valid_c=12),
        Uop(UOP_STORE, flags=cflag, src0=LS_B, dst=6, in_h=128, in_w=256, in_c=12, out_c=64, c_offset=28, valid_c=12),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=11, in_h=128, in_w=256, in_c=12, out_c=12, kernel=3, stride=1, dilation=8, padding=8, valid_c=12),
        Uop(UOP_ADD, flags=addflag, src0=LS_B, src1=LS_TMP, dst=LS_A, param_id=4, in_h=128, in_w=256, in_c=12, out_c=12, valid_c=12),
        Uop(UOP_STORE, flags=cflag, src0=LS_A, dst=6, in_h=128, in_w=256, in_c=12, out_c=64, c_offset=40, valid_c=12),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=12, in_h=128, in_w=256, in_c=12, out_c=12, kernel=3, stride=1, dilation=16, padding=16, valid_c=12),
        Uop(UOP_ADD, flags=addflag, src0=LS_A, src1=LS_TMP, dst=LS_B, param_id=5, in_h=128, in_w=256, in_c=12, out_c=12, valid_c=12),
        Uop(UOP_STORE, flags=cflag, src0=LS_B, dst=6, in_h=128, in_w=256, in_c=12, out_c=64, c_offset=52, valid_c=12),
        Uop(UOP_ADD, flags=addflag, src0=6, src1=5, dst=6, param_id=6, in_h=128, in_w=256, in_c=64, out_c=64, valid_c=64),
        stage_last(Uop(UOP_AFFINE, flags=aflag, src0=6, dst=7, param_id=2, act_type=ACT_RELU, in_h=128, in_w=256, in_c=64, out_c=64, valid_c=64)),
        Uop(UOP_STORE, flags=cflag, src0=7, dst=9, in_h=128, in_w=256, in_c=64, out_c=131, c_offset=0, valid_c=64),
        Uop(UOP_STORE, flags=cflag, src0=5, dst=9, in_h=128, in_w=256, in_c=64, out_c=131, c_offset=64, valid_c=64),
        Uop(UOP_STORE, flags=cflag, src0=8, dst=9, in_h=128, in_w=256, in_c=3, out_c=131, c_offset=128, valid_c=3),
        stage_last(Uop(UOP_AFFINE, flags=aflag, src0=9, dst=10, param_id=3, act_type=ACT_RELU, in_h=128, in_w=256, in_c=131, out_c=131, valid_c=131)),
        Uop(UOP_CONV, src0=10, dst=LS_C1, param_id=13, in_h=128, in_w=256, in_c=131, out_c=25, kernel=3, stride=2, dilation=1, padding=1, valid_c=25),
        Uop(UOP_CONV, flags=cflag, src0=LS_C1, dst=11, param_id=14, in_h=64, in_w=128, in_c=25, out_c=28, kernel=3, stride=1, dilation=1, padding=1, c_offset=0, valid_c=28),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_A, param_id=15, in_h=64, in_w=128, in_c=25, out_c=25, kernel=3, stride=1, dilation=2, padding=2, valid_c=25),
        Uop(UOP_STORE, flags=cflag, src0=LS_A, dst=11, in_h=64, in_w=128, in_c=25, out_c=128, c_offset=28, valid_c=25),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=16, in_h=64, in_w=128, in_c=25, out_c=25, kernel=3, stride=1, dilation=4, padding=4, valid_c=25),
        Uop(UOP_ADD, flags=addflag, src0=LS_A, src1=LS_TMP, dst=LS_B, param_id=7, in_h=64, in_w=128, in_c=25, out_c=25, valid_c=25),
        Uop(UOP_STORE, flags=cflag, src0=LS_B, dst=11, in_h=64, in_w=128, in_c=25, out_c=128, c_offset=53, valid_c=25),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=17, in_h=64, in_w=128, in_c=25, out_c=25, kernel=3, stride=1, dilation=8, padding=8, valid_c=25),
        Uop(UOP_ADD, flags=addflag, src0=LS_B, src1=LS_TMP, dst=LS_A, param_id=8, in_h=64, in_w=128, in_c=25, out_c=25, valid_c=25),
        Uop(UOP_STORE, flags=cflag, src0=LS_A, dst=11, in_h=64, in_w=128, in_c=25, out_c=128, c_offset=78, valid_c=25),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=18, in_h=64, in_w=128, in_c=25, out_c=25, kernel=3, stride=1, dilation=16, padding=16, valid_c=25),
        Uop(UOP_ADD, flags=addflag, src0=LS_A, src1=LS_TMP, dst=LS_B, param_id=9, in_h=64, in_w=128, in_c=25, out_c=25, valid_c=25),
        Uop(UOP_STORE, flags=cflag, src0=LS_B, dst=11, in_h=64, in_w=128, in_c=25, out_c=128, c_offset=103, valid_c=25),
        stage_last(Uop(UOP_AFFINE, flags=aflag, src0=11, dst=12, param_id=4, act_type=ACT_RELU, in_h=64, in_w=128, in_c=128, out_c=128, valid_c=128)),
        Uop(UOP_CONV, src0=12, dst=LS_C1, param_id=19, in_h=64, in_w=128, in_c=128, out_c=25, kernel=1, stride=1, dilation=1, padding=0, valid_c=25),
        Uop(UOP_CONV, flags=cflag, src0=LS_C1, dst=13, param_id=20, in_h=64, in_w=128, in_c=25, out_c=28, kernel=3, stride=1, dilation=1, padding=1, c_offset=0, valid_c=28),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_A, param_id=21, in_h=64, in_w=128, in_c=25, out_c=25, kernel=3, stride=1, dilation=2, padding=2, valid_c=25),
        Uop(UOP_STORE, flags=cflag, src0=LS_A, dst=13, in_h=64, in_w=128, in_c=25, out_c=128, c_offset=28, valid_c=25),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=22, in_h=64, in_w=128, in_c=25, out_c=25, kernel=3, stride=1, dilation=4, padding=4, valid_c=25),
        Uop(UOP_ADD, flags=addflag, src0=LS_A, src1=LS_TMP, dst=LS_B, param_id=10, in_h=64, in_w=128, in_c=25, out_c=25, valid_c=25),
        Uop(UOP_STORE, flags=cflag, src0=LS_B, dst=13, in_h=64, in_w=128, in_c=25, out_c=128, c_offset=53, valid_c=25),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=23, in_h=64, in_w=128, in_c=25, out_c=25, kernel=3, stride=1, dilation=8, padding=8, valid_c=25),
        Uop(UOP_ADD, flags=addflag, src0=LS_B, src1=LS_TMP, dst=LS_A, param_id=11, in_h=64, in_w=128, in_c=25, out_c=25, valid_c=25),
        Uop(UOP_STORE, flags=cflag, src0=LS_A, dst=13, in_h=64, in_w=128, in_c=25, out_c=128, c_offset=78, valid_c=25),
        Uop(UOP_CONV, src0=LS_C1, dst=LS_TMP, param_id=24, in_h=64, in_w=128, in_c=25, out_c=25, kernel=3, stride=1, dilation=16, padding=16, valid_c=25),
        Uop(UOP_ADD, flags=addflag, src0=LS_A, src1=LS_TMP, dst=LS_B, param_id=12, in_h=64, in_w=128, in_c=25, out_c=25, valid_c=25),
        Uop(UOP_STORE, flags=cflag, src0=LS_B, dst=13, in_h=64, in_w=128, in_c=25, out_c=128, c_offset=103, valid_c=25),
        Uop(UOP_ADD, flags=addflag, src0=13, src1=12, dst=13, param_id=13, in_h=64, in_w=128, in_c=128, out_c=128, valid_c=128),
        stage_last(Uop(UOP_AFFINE, flags=aflag, src0=13, dst=14, param_id=5, act_type=ACT_RELU, in_h=64, in_w=128, in_c=128, out_c=128, valid_c=128)),
        Uop(UOP_STORE, flags=cflag, src0=12, dst=15, in_h=64, in_w=128, in_c=128, out_c=256, c_offset=0, valid_c=128),
        Uop(UOP_STORE, flags=cflag, src0=14, dst=15, in_h=64, in_w=128, in_c=128, out_c=256, c_offset=128, valid_c=128),
        Uop(UOP_AFFINE, flags=aflag, src0=15, dst=16, param_id=6, act_type=ACT_RELU, in_h=64, in_w=128, in_c=256, out_c=256, valid_c=256),
        Uop(UOP_CONV, src0=16, dst=17, param_id=25, in_h=64, in_w=128, in_c=256, out_c=2, kernel=1, stride=1, dilation=1, padding=0, valid_c=2),
        Uop(UOP_STORE, src0=17, in_h=64, in_w=128, in_c=2, out_c=2, valid_c=2),
        stage_last(Uop(UOP_END)),
    ]

    if len(uops) != 75:
        raise AssertionError(f"Expected 75 uops, got {len(uops)}")
    return uops


def build_window_schedule_sections(uops: Sequence[Uop]) -> Tuple[List[WindowSchedDesc], List[WindowPackCmd], Dict[int, int], List[dict]]:
    schedules: List[WindowSchedDesc] = []
    commands: List[WindowPackCmd] = []
    schedule_ids_by_param: Dict[int, int] = {}
    cache: Dict[Tuple[int, int, int], int] = {}

    for uop in uops:
        if uop.opcode != UOP_CONV:
            continue
        source_aligned = tensor_supports_aligned_1x1_read(uop.src0)
        mode = window_mode_for(uop.in_c, uop.kernel, source_aligned)
        key = (uop.kernel, uop.in_c, mode)
        if key not in cache:
            local_desc, local_cmds = make_window_pack_schedule(uop.in_c, uop.kernel, mode=mode)
            if len(commands) + len(local_cmds) > MAX_WINDOW_PACK_CMD_COUNT:
                raise ValueError("window_pack_cmd_count exceeds MAX_WINDOW_PACK_CMD_COUNT")
            global_desc = WindowSchedDesc(
                mode=local_desc.mode,
                kernel=local_desc.kernel,
                stride=0,
                dilation=0,
                in_c=local_desc.in_c,
                k_tiles=local_desc.k_tiles,
                cmd_base=len(commands),
                cmd_count=len(local_cmds),
                kt_cmd_base=local_desc.kt_cmd_base,
                flags=0,
            )
            cache[key] = len(schedules)
            schedules.append(global_desc)
            commands.extend(local_cmds)
        schedule_ids_by_param[uop.param_id] = cache[key]

    audit = []
    for idx, desc in enumerate(schedules):
        cmds_per_kt = [
            desc.kt_cmd_base[kt + 1] - desc.kt_cmd_base[kt]
            for kt in range(desc.k_tiles)
        ]
        byte_count_by_kt = [
            sum(commands[desc.cmd_base + j].byte_count for j in range(desc.kt_cmd_base[kt], desc.kt_cmd_base[kt + 1]))
            for kt in range(desc.k_tiles)
        ]
        lane_level_suspected = desc.cmd_count >= desc.k_tiles * 16 or sum(
            1 for j in range(desc.cmd_base, desc.cmd_base + desc.cmd_count)
            if commands[j].byte_count == 1
        ) > desc.k_tiles * 8
        audit.append(
            {
                "id": idx,
                "key": [desc.kernel, desc.in_c, desc.mode],
                "k_tiles": desc.k_tiles,
                "cmd_base": desc.cmd_base,
                "cmd_count": desc.cmd_count,
                "cmds_per_kt": cmds_per_kt,
                "bytes_per_kt": byte_count_by_kt,
                "lane_level_suspected": lane_level_suspected,
            }
        )
    return schedules, commands, schedule_ids_by_param, audit


def _uop_is_store_of(uop: Uop, src: int) -> bool:
    return uop.opcode == UOP_STORE and uop.src0 == src


def _uop_is_add_of(uop: Uop, src: int) -> bool:
    return uop.opcode == UOP_ADD and (uop.src0 == src or uop.src1 == src)


def build_exec_plan_sections(
    uops: Sequence[Uop],
    schedule_ids_by_param: Dict[int, int],
    packed_weight_word_offsets: Dict[int, int],
    weight_report: Sequence[dict],
) -> Tuple[List[ConvExecDesc], List[RowConsumerDesc], List[ExecPlanEntry], dict]:
    weight_words_by_param = {int(x["param_id"]): int(x["packed_words"]) for x in weight_report}
    k_tiles_by_param = {int(x["param_id"]): int(x["k_tiles"]) for x in weight_report}
    conv_exec_descs: List[ConvExecDesc] = []
    row_consumers: List[RowConsumerDesc] = []
    exec_plan: List[ExecPlanEntry] = []
    skipped: set[int] = set()

    add_total = sum(1 for u in uops if u.opcode == UOP_ADD)
    store_total = sum(1 for u in uops if u.opcode == UOP_STORE)
    affine_total = sum(1 for u in uops if u.opcode == UOP_AFFINE)
    add_fused = 0
    store_as_row_consumer = 0
    standalone_store_fixed = 0
    affine_fused = 0
    standalone_affine = 0

    for idx, uop in enumerate(uops):
        if idx in skipped or uop.opcode in (UOP_LOAD_FM, UOP_NOP):
            continue
        if uop.opcode == UOP_END:
            break

        if uop.opcode == UOP_CONV:
            out_w = conv_out_dim(uop.in_w, uop.stride)
            row_consumer = RowConsumerDesc(
                mode=ROW_CONSUMER_NONE,
                store_dst_tensor=uop.dst,
                store_c_offset=uop.c_offset,
                valid_c=uop.valid_c,
                reserved0=store_layout_for(
                    dst_tensor=uop.dst,
                    producer=uop,
                    c_offset=uop.c_offset,
                    valid_c=uop.valid_c,
                    out_w=out_w,
                ),
            )
            if idx + 1 < len(uops) and _uop_is_store_of(uops[idx + 1], uop.dst):
                store = uops[idx + 1]
                mode = ROW_CONSUMER_UPSAMPLE_OUT if store.dst == TID_INVALID else ROW_CONSUMER_STORE
                layout = (
                    STORE_LAYOUT_NONE
                    if mode == ROW_CONSUMER_UPSAMPLE_OUT
                    else store_layout_for(
                        dst_tensor=store.dst,
                        producer=uop,
                        c_offset=store.c_offset,
                        valid_c=store.valid_c,
                        out_w=out_w,
                    )
                )
                row_consumer = RowConsumerDesc(
                    mode=mode,
                    store_dst_tensor=store.dst,
                    store_c_offset=store.c_offset,
                    valid_c=store.valid_c,
                    alias_tensor=(uop.dst if mode == ROW_CONSUMER_STORE and uop.dst != store.dst else TID_INVALID),
                    reserved0=layout,
                )
                skipped.add(idx + 1)
                store_as_row_consumer += 1
            elif (
                idx + 2 < len(uops)
                and _uop_is_add_of(uops[idx + 1], uop.dst)
                and _uop_is_store_of(uops[idx + 2], uops[idx + 1].dst)
            ):
                add = uops[idx + 1]
                store = uops[idx + 2]
                other = add.src1 if add.src0 == uop.dst else add.src0
                row_consumer = RowConsumerDesc(
                    mode=ROW_CONSUMER_ADD_STORE,
                    add_other_tensor=other,
                    store_dst_tensor=store.dst,
                    add_qparam_id=add.param_id,
                    store_c_offset=store.c_offset,
                    valid_c=store.valid_c,
                    alias_tensor=(add.dst if add.dst != store.dst else TID_INVALID),
                    reserved0=store_layout_for(
                        dst_tensor=store.dst,
                        producer=uop,
                        c_offset=store.c_offset,
                        valid_c=store.valid_c,
                        out_w=out_w,
                    ),
                )
                skipped.update({idx + 1, idx + 2})
                add_fused += 1
                store_as_row_consumer += 1

            row_id = len(row_consumers)
            row_consumers.append(row_consumer)
            conv_id = len(conv_exec_descs)
            conv_exec_descs.append(
                ConvExecDesc(
                    param_id=uop.param_id,
                    qparam_id=uop.qparam_id if uop.qparam_id else uop.param_id,
                    window_sched_id=schedule_ids_by_param[uop.param_id],
                    row_consumer_id=row_id,
                    in_h=uop.in_h,
                    in_w=uop.in_w,
                    in_c=uop.in_c,
                    out_c=uop.out_c,
                    kernel=uop.kernel,
                    stride=uop.stride,
                    dilation=uop.dilation,
                    padding=uop.padding,
                    src_tensor=uop.src0,
                    dst_tensor=uop.dst,
                    dst_c_offset=uop.c_offset,
                    valid_c=uop.valid_c,
                    packed_weight_word_offset=packed_weight_word_offsets[uop.param_id],
                    k_tiles=k_tiles_by_param[uop.param_id],
                    weight_words=weight_words_by_param[uop.param_id],
                    flags=uop.flags,
                )
            )
            exec_plan.append(ExecPlanEntry(EXEC_CONV, conv_id, idx, 0))
            continue

        if uop.opcode == UOP_POOL:
            exec_plan.append(ExecPlanEntry(EXEC_POOL, uop.param_id, idx, uop.flags))
            continue

        if uop.opcode == UOP_AFFINE:
            standalone_affine += 1
            exec_plan.append(ExecPlanEntry(EXEC_AFFINE, uop.param_id, idx, uop.flags))
            continue

        if uop.opcode == UOP_STORE:
            standalone_store_fixed += 1
            exec_plan.append(ExecPlanEntry(EXEC_STORE, idx, idx, uop.flags))
            continue

        if uop.opcode == UOP_ADD:
            if idx + 1 < len(uops) and uops[idx + 1].opcode == UOP_AFFINE and uops[idx + 1].src0 == uop.dst:
                affine = uops[idx + 1]
                exec_plan.append(ExecPlanEntry(EXEC_ADD_AFFINE, affine.param_id, idx + 1, affine.flags))
                skipped.add(idx + 1)
                add_fused += 1
                affine_fused += 1
                continue
            raise ValueError(f"standalone ADD uop[{idx}] is not covered by P7 fusion plan")

        raise ValueError(f"unsupported logical uop[{idx}] opcode={uop.opcode}")

    exec_plan.append(ExecPlanEntry(EXEC_END, 0, len(uops) - 1, 0))
    if len(exec_plan) > MAX_EXEC_PLAN_COUNT:
        raise ValueError(f"exec_plan_count={len(exec_plan)} exceeds MAX_EXEC_PLAN_COUNT={MAX_EXEC_PLAN_COUNT}")
    coverage = {
        "legacy_uop_count": len(uops),
        "exec_entry_count": len(exec_plan),
        "conv_count": len(conv_exec_descs),
        "add_total": add_total,
        "add_fused": add_fused,
        "standalone_add": add_total - add_fused,
        "store_total": store_total,
        "store_as_row_consumer": store_as_row_consumer,
        "standalone_store_fixed": standalone_store_fixed,
        "standalone_concat": 0,
        "affine_total": affine_total,
        "affine_fused": affine_fused,
        "standalone_affine": standalone_affine,
        "store_layouts": {
            STORE_LAYOUT_NAMES.get(layout, str(layout)): sum(1 for rc in row_consumers if rc.reserved0 == layout)
            for layout in sorted({rc.reserved0 for rc in row_consumers})
        },
    }
    if coverage["standalone_add"] != 0:
        raise ValueError(f"ADD fusion incomplete: {coverage}")
    return conv_exec_descs, row_consumers, exec_plan, coverage


def build_hw_frame_files(artifact_dir: Path, out_dir: Path) -> Dict[str, object]:
    golden_dir = artifact_dir / "golden_sample"
    input_nchw = require_int8_array(golden_dir / "quant" / "input_int.npy")
    output_nchw = require_int8_array(golden_dir / "classifier" / "output_int.npy")

    if input_nchw.shape != (1, 3, 512, 1024):
        raise ValueError(f"Unexpected input shape: {input_nchw.shape}")
    if output_nchw.shape != (1, 2, 64, 128):
        raise ValueError(f"Unexpected output shape: {output_nchw.shape}")

    input_nhwc = np.ascontiguousarray(np.transpose(input_nchw, (0, 2, 3, 1)))
    output_nhwc = np.ascontiguousarray(np.transpose(output_nchw, (0, 2, 3, 1)))

    np.save(out_dir / "input_q_nhwc.npy", input_nhwc)
    np.save(out_dir / "golden_output_q_nhwc.npy", output_nhwc)
    input_nhwc.tofile(out_dir / "input_q.bin")
    input_nhwc.tofile(out_dir / "INPUTQ.BIN")
    output_nhwc.tofile(out_dir / "golden_output_q.bin")
    output_nhwc.tofile(out_dir / "expected_output_q.bin")

    fp32_src = golden_dir / "model_output_fp32.npy"
    target_src = golden_dir / "target.npy"
    sample_info_src = golden_dir / "sample_info.json"
    if fp32_src.exists():
        shutil.copy2(fp32_src, out_dir / "golden_output_fp32.npy")
    if target_src.exists():
        shutil.copy2(target_src, out_dir / "target.npy")
    if sample_info_src.exists():
        shutil.copy2(sample_info_src, out_dir / "sample_info.json")

    return {
        "input_shape_nhwc": list(input_nhwc.shape),
        "output_shape_nhwc": list(output_nhwc.shape),
        "input_q_bin_bytes": int(input_nhwc.nbytes),
        "golden_output_q_bin_bytes": int(output_nhwc.nbytes),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_param_audit(
    *,
    qparams: Dict[str, Tuple[float, int, str]],
    scale_names: Sequence[str],
    section_offsets: Dict[str, int],
    section_sizes: Dict[str, int],
    window_audit: Sequence[dict],
    weight_report: Sequence[dict],
    fusion_coverage: dict,
    warnings: Sequence[str],
) -> dict:
    required_smallc = {
        (3, uop.in_c)
        for uop in build_uops()
        if uop.opcode == UOP_CONV and uop.kernel == 3 and 3 < uop.in_c <= 32
    }
    present = {(int(x["key"][0]), int(x["key"][1])) for x in window_audit}
    missing = sorted(list(required_smallc - present))
    if missing:
        raise ValueError(f"missing required small-C window schedules: {missing}")
    if any(x["lane_level_suspected"] for x in window_audit):
        raise ValueError("lane-level window schedule suspected; refusing PARAM v3 export")
    if fusion_coverage["add_fused"] != fusion_coverage["add_total"]:
        raise ValueError(f"ADD fusion incomplete: {fusion_coverage}")
    store_layouts = fusion_coverage.get("store_layouts", {})
    if store_layouts.get(STORE_LAYOUT_NAMES[STORE_LAYOUT_COLD_RMW_FALLBACK], 0):
        raise ValueError(f"S5 hot path selected cold RMW fallback: {store_layouts}")

    scale_report = [
        {
            "scale_id": idx,
            "name": name,
            "scale": qparams[name][0],
            "zero_point": qparams[name][1],
            "source": qparams[name][2],
        }
        for idx, name in enumerate(scale_names)
    ]
    return {
        "format": "ESP_INT8_PARAM_AUDIT_V3",
        "param_version": PARAM_BLOB_VERSION_SCHED,
        "section_offsets": section_offsets,
        "section_sizes": section_sizes,
        "scale_table": scale_report,
        "window_schedules": list(window_audit),
        "window_pack_cmd_count": sum(int(x["cmd_count"]) for x in window_audit),
        "window_pack_cmd_bytes": sum(int(x["cmd_count"]) for x in window_audit) * 8,
        "packed_weights": {
            "total_bytes": sum(int(x["packed_words"]) for x in weight_report) * TK,
            "wbuf_bytes": WBUF_BYTES,
            "layers": list(weight_report),
        },
        "fusion_coverage": fusion_coverage,
        "hard_checks": {
            "required_smallc_schedules_present": True,
            "no_lane_level_window_schedule": True,
            "all_adds_fused": True,
            "no_standalone_concat": fusion_coverage["standalone_concat"] == 0,
            "packed_weight_fits_wbuf": sum(int(x["packed_words"]) for x in weight_report) * TK <= WBUF_BYTES,
            "store_layouts_scheduled": bool(store_layouts),
            "no_store_cold_rmw_fallback": store_layouts.get(STORE_LAYOUT_NAMES[STORE_LAYOUT_COLD_RMW_FALLBACK], 0) == 0,
        },
        "warnings": list(warnings),
    }


def build_blob_v3(args: argparse.Namespace) -> dict:
    artifact_dir = Path(args.artifact_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    qparams = collect_activation_qparams(artifact_dir)
    scale_names, scale_ids = build_scale_table(qparams)
    for name in scale_names:
        scale, zp, source = qparams[name]
        if zp != 0:
            warnings.append(
                f"{name}: activation zero_point={zp} from {source}; P7 hardware expects symmetric activations."
            )

    state = torch.load(artifact_dir / "fake_quant_state_dict.pth", map_location="cpu")
    packed_weight_blob, packed_weight_word_offsets, weight_report = build_packed_weight_section(artifact_dir, warnings)
    conv_q_blob, conv_q_local_offsets, conv_report = build_conv_qparams(artifact_dir, state, qparams)
    affine_q_blob, affine_q_local_offsets, affine_report = build_affine_qparams(state, qparams)
    add_q_blob, add_q_local_offsets, add_report = build_add_qparams(qparams, scale_ids, warnings)
    pool_q_blob, pool_q_local_offsets, pool_report = build_pool_qparams(qparams, scale_ids)

    uops = build_uops()
    window_scheds, window_cmds, schedule_ids_by_param, window_audit = build_window_schedule_sections(uops)
    conv_exec_descs, row_consumers, exec_plan, fusion_coverage = build_exec_plan_sections(
        uops,
        schedule_ids_by_param,
        packed_weight_word_offsets,
        weight_report,
    )

    tensor_desc_data = b"".join(pack_tensor_desc(desc) for desc in TENSOR_PLAN)
    scale_desc_data = b"".join(pack_scale_desc(qparams[name][0]) for name in scale_names)
    conv_exec_data = b"".join(pack_conv_exec_desc(desc) for desc in conv_exec_descs)
    window_sched_data = b"".join(pack_window_sched_desc(desc) for desc in window_scheds)
    window_cmd_data = b"".join(pack_window_pack_cmd(cmd) for cmd in window_cmds)
    row_consumer_data = b"".join(pack_row_consumer_desc(desc) for desc in row_consumers)
    exec_plan_data = b"".join(pack_exec_plan_entry(entry) for entry in exec_plan)
    uop_data = b"".join(pack_uop(uop) for uop in uops)

    section_data = {
        "tensor_desc": tensor_desc_data,
        "scale_desc": scale_desc_data,
        "conv_exec_desc": conv_exec_data,
        "window_sched_desc": window_sched_data,
        "window_pack_cmd": window_cmd_data,
        "row_consumer_desc": row_consumer_data,
        "exec_plan": exec_plan_data,
        "weight_packed": packed_weight_blob,
        "conv_qparam": conv_q_blob,
        "affine_qparam": affine_q_blob,
        "add_qparam": add_q_blob,
        "pool_qparam": pool_q_blob,
        "legacy_uop_debug": uop_data,
    }

    offsets: Dict[str, int] = {}
    cursor = PARAM_HEADER_BYTES
    for name, data in section_data.items():
        cursor = align_up(cursor)
        offsets[name] = cursor
        cursor += len(data)

    section_sizes = {"header": PARAM_HEADER_BYTES, **{name: len(data) for name, data in section_data.items()}}
    header_words = [
        PARAM_BLOB_MAGIC,
        PARAM_BLOB_VERSION_SCHED,
        len(TENSOR_PLAN),
        len(scale_names),
        len(conv_exec_descs),
        len(AFFINE_PLAN),
        len(ADD_PLAN),
        len(POOL_PLAN),
        len(uops),
        len(exec_plan),
        offsets["tensor_desc"],
        offsets["scale_desc"],
        0,
        0,
        0,
        0,
        offsets["legacy_uop_debug"],
        offsets["weight_packed"],
        offsets["conv_qparam"],
        offsets["affine_qparam"],
        offsets["add_qparam"],
        offsets["pool_qparam"],
        offsets["conv_exec_desc"],
        offsets["window_sched_desc"],
        offsets["window_pack_cmd"],
        offsets["row_consumer_desc"],
        0,
        offsets["exec_plan"],
        len(window_scheds),
        len(window_cmds),
        len(conv_exec_descs),
        len(row_consumers),
    ]

    blob = bytearray(struct.pack("<32I", *header_words))

    def append_section(name: str, data: bytes) -> None:
        if len(blob) > offsets[name]:
            raise AssertionError(f"Section overlap before {name}")
        blob.extend(b"\x00" * (offsets[name] - len(blob)))
        blob.extend(data)

    for name, data in section_data.items():
        append_section(name, data)
    pad_to(blob)

    audit = build_param_audit(
        qparams=qparams,
        scale_names=scale_names,
        section_offsets=offsets,
        section_sizes=section_sizes,
        window_audit=window_audit,
        weight_report=weight_report,
        fusion_coverage=fusion_coverage,
        warnings=warnings,
    )

    blob_path = out_dir / "param_blob.bin"
    blob_path.write_bytes(blob)
    shutil.copy2(blob_path, out_dir / "PARAM.BIN")
    (out_dir / "uop_table.bin").write_bytes(uop_data)
    (out_dir / "tensor_desc_table.bin").write_bytes(tensor_desc_data)
    (out_dir / "exec_plan.bin").write_bytes(exec_plan_data)
    (out_dir / "window_pack_cmd.bin").write_bytes(window_cmd_data)

    frame_report = build_hw_frame_files(artifact_dir, out_dir)

    scale_report = audit["scale_table"]
    (out_dir / "scale_table.json").write_text(
        json.dumps(scale_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    audit["param_blob"] = {
        "file": "PARAM.BIN",
        "param_blob_file": "param_blob.bin",
        "bytes": len(blob),
        "sha256": sha256_file(blob_path),
    }
    (out_dir / "param_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "format": "ESP_INT8_HW_ARTIFACT_SINGLE_V3",
        "artifact_dir": str(artifact_dir),
        "param_blob": str(blob_path),
        "param_bin": str(out_dir / "PARAM.BIN"),
        "param_blob_bytes": len(blob),
        "param_blob_sha256": audit["param_blob"]["sha256"],
        "param_version": PARAM_BLOB_VERSION_SCHED,
        "hardware_contract": {
            "memory_layout": "shared_fmbuf_compact_l20_view_20260511",
            "schedule": "P7_PARAM_V3_EXEC_PLAN",
            "weight_layout": "SA-ready packed weight words: oc-local TM lane, kt K-tile",
            "window_schedule": "segment-level K tile pack commands, not per-lane/per-pixel absolute addresses",
            "board_input": "NHWC signed int8, 512x1024x3",
            "board_output": "full-res uint8 mask for P7 fullres hardware, or low-res logits debug files",
        },
        "uop_count": len(uops),
        "exec_entry_count": len(exec_plan),
        "tensor_desc_count": len(TENSOR_PLAN),
        "scale_desc_count": len(scale_names),
        "conv_exec_desc_count": len(conv_exec_descs),
        "window_sched_count": len(window_scheds),
        "window_pack_cmd_count": len(window_cmds),
        "row_consumer_desc_count": len(row_consumers),
        "section_offsets": offsets,
        "section_sizes": section_sizes,
        "frame_data": frame_report,
        "conv_qparams": conv_report,
        "affine_qparams": affine_report,
        "add_qparams": add_report,
        "pool_qparams": pool_report,
        "fusion_coverage": fusion_coverage,
        "warnings": warnings,
        "param_audit": str(out_dir / "param_audit.json"),
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "single_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.strict_zp and warnings:
        raise RuntimeError("Strict zero-point/alignment mode failed; see export_manifest.json warnings.")
    if args.strict_schedule:
        parse_and_check_blob_v3(blob_path, audit)
    return manifest


def build_blob_v1(args: argparse.Namespace) -> dict:
    artifact_dir = Path(args.artifact_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    qparams = collect_activation_qparams(artifact_dir)
    scale_names, scale_ids = build_scale_table(qparams)

    for name in scale_names:
        scale, zp, source = qparams[name]
        if zp != 0:
            warnings.append(
                f"{name}: activation zero_point={zp} from {source}; current hardware design is optimized for zp=0."
            )

    state = torch.load(artifact_dir / "fake_quant_state_dict.pth", map_location="cpu")

    weight_blob, weight_local_offsets = build_weight_section(artifact_dir, warnings)
    conv_q_blob, conv_q_local_offsets, conv_report = build_conv_qparams(artifact_dir, state, qparams)
    affine_q_blob, affine_q_local_offsets, affine_report = build_affine_qparams(state, qparams)
    add_q_blob, add_q_local_offsets, add_report = build_add_qparams(qparams, scale_ids, warnings)
    pool_q_blob, pool_q_local_offsets, pool_report = build_pool_qparams(qparams, scale_ids)
    uops = build_uops()

    section_sizes = {
        "header": 128,
        "tensor_desc": 16 * len(TENSORS),
        "scale_desc": 8 * len(scale_names),
        "conv_desc": 16 * len(CONVS),
        "affine_desc": 16 * len(AFFINES),
        "add_desc": 16 * len(ADDS),
        "pool_desc": 16 * len(POOLS),
        "uop": 32 * len(uops),
        "weight_data": len(weight_blob),
        "conv_qparam": len(conv_q_blob),
        "affine_qparam": len(affine_q_blob),
        "add_qparam": len(add_q_blob),
        "pool_qparam": len(pool_q_blob),
    }

    offsets: Dict[str, int] = {}
    cursor = 128
    for name in [
        "tensor_desc",
        "scale_desc",
        "conv_desc",
        "affine_desc",
        "add_desc",
        "pool_desc",
        "uop",
        "weight_data",
        "conv_qparam",
        "affine_qparam",
        "add_qparam",
        "pool_qparam",
    ]:
        cursor = align_up(cursor)
        offsets[name] = cursor
        cursor += section_sizes[name]

    blob = bytearray()
    header_words = [
        PARAM_BLOB_MAGIC,
        PARAM_BLOB_VERSION,
        len(TENSORS),
        len(scale_names),
        len(CONVS),
        len(AFFINES),
        len(ADDS),
        len(POOLS),
        len(uops),
        0,
        offsets["tensor_desc"],
        offsets["scale_desc"],
        offsets["conv_desc"],
        offsets["affine_desc"],
        offsets["add_desc"],
        offsets["pool_desc"],
        offsets["uop"],
        offsets["weight_data"],
        offsets["conv_qparam"],
        offsets["affine_qparam"],
        offsets["add_qparam"],
        offsets["pool_qparam"],
        *([0] * 10),
    ]
    blob.extend(struct.pack("<32I", *header_words))

    def append_section(name: str, data: bytes) -> None:
        if len(blob) > offsets[name]:
            raise AssertionError(f"Section overlap before {name}")
        blob.extend(b"\x00" * (offsets[name] - len(blob)))
        blob.extend(data)

    tensor_desc_data = b"".join(pack_tensor_desc(desc) for desc in TENSORS)
    scale_desc_data = b"".join(pack_scale_desc(qparams[name][0]) for name in scale_names)
    conv_desc_data = b"".join(
        pack_param_desc(
            offsets["weight_data"] + weight_local_offsets[spec.param_id],
            offsets["conv_qparam"] + conv_q_local_offsets[spec.param_id],
            offsets["conv_qparam"] + conv_q_local_offsets[spec.param_id],
        )
        for spec in CONVS
    )
    affine_desc_data = b"".join(
        pack_param_desc(offsets["affine_qparam"] + affine_q_local_offsets[spec.param_id])
        for spec in AFFINES
    )
    add_desc_data = b"".join(
        pack_param_desc(offsets["add_qparam"] + add_q_local_offsets[spec.param_id])
        for spec in ADDS
    )
    pool_desc_data = b"".join(
        pack_param_desc(offsets["pool_qparam"] + pool_q_local_offsets[spec.param_id])
        for spec in POOLS
    )
    uop_data = b"".join(pack_uop(uop) for uop in uops)

    append_section("tensor_desc", tensor_desc_data)
    append_section("scale_desc", scale_desc_data)
    append_section("conv_desc", conv_desc_data)
    append_section("affine_desc", affine_desc_data)
    append_section("add_desc", add_desc_data)
    append_section("pool_desc", pool_desc_data)
    append_section("uop", uop_data)
    append_section("weight_data", weight_blob)
    append_section("conv_qparam", conv_q_blob)
    append_section("affine_qparam", affine_q_blob)
    append_section("add_qparam", add_q_blob)
    append_section("pool_qparam", pool_q_blob)
    pad_to(blob)

    blob_path = out_dir / "param_blob.bin"
    blob_path.write_bytes(blob)
    (out_dir / "uop_table.bin").write_bytes(uop_data)
    (out_dir / "tensor_desc_table.bin").write_bytes(tensor_desc_data)

    frame_report = build_hw_frame_files(artifact_dir, out_dir)

    scale_report = [
        {
            "scale_id": idx,
            "name": name,
            "scale": qparams[name][0],
            "zero_point": qparams[name][1],
            "source": qparams[name][2],
        }
        for idx, name in enumerate(scale_names)
    ]
    (out_dir / "scale_table.json").write_text(
        json.dumps(scale_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "artifact_dir": str(artifact_dir),
        "param_blob": str(blob_path),
        "param_blob_bytes": len(blob),
        "hardware_contract": {
            "memory_layout": "shared_fmbuf_compact_l20_view_20260511",
            "tensor_desc_reserved0": "physical_c_stride",
            "tensor_desc_reserved1": "channel_offset",
            "pool2_schedule": "uop_5_6_before_level2_compute",
            "view_tensors": [
                "T_L20_CAT",
                "T_L20_ACT",
                "T_L2B0_CAT",
                "T_L2B0_ACT",
                "T_L3B0_CAT",
                "T_L3B0_ACT",
            ],
        },
        "uop_count": len(uops),
        "tensor_desc_count": len(TENSORS),
        "scale_desc_count": len(scale_names),
        "conv_desc_count": len(CONVS),
        "affine_desc_count": len(AFFINES),
        "add_desc_count": len(ADDS),
        "pool_desc_count": len(POOLS),
        "section_offsets": offsets,
        "section_sizes": section_sizes,
        "frame_data": frame_report,
        "conv_qparams": conv_report,
        "affine_qparams": affine_report,
        "add_qparams": add_report,
        "pool_qparams": pool_report,
        "warnings": warnings,
        "note": (
            "This export uses the single golden_sample generated by "
            "D:\\ESPNet\\export_quantized_artifacts.py, not the full validation set."
        ),
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.strict_zp and warnings:
        raise RuntimeError("Strict zero-point/alignment mode failed; see export_manifest.json warnings.")

    return manifest


def parse_and_check_blob(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 128:
        raise ValueError("param_blob.bin is shorter than 128-byte header")
    header = struct.unpack("<32I", data[:128])
    if header[0] != PARAM_BLOB_MAGIC or header[1] != PARAM_BLOB_VERSION:
        raise ValueError("bad param blob magic/version")
    for off in header[10:22]:
        if off % SECTION_ALIGN != 0:
            raise ValueError(f"unaligned section offset: {off}")
        if off >= len(data):
            raise ValueError(f"section offset out of range: {off}")
    tensor_count = header[2]
    tensor_off = header[10]
    if tensor_count != len(TENSORS):
        raise ValueError(f"unexpected tensor_desc_count {tensor_count}")
    for idx, expected in enumerate(TENSORS):
        off = tensor_off + idx * 16
        got = struct.unpack("<BBHIHHHH", data[off : off + 16])
        expected_tuple = (
            expected.bank_id,
            1,
            expected.phys_c,
            expected.base_offset,
            expected.h,
            expected.w,
            expected.c,
            expected.c_offset,
        )
        if got != expected_tuple:
            raise ValueError(f"tensor_desc[{idx}] {expected.name} mismatch: got={got}, expected={expected_tuple}")

    uop_count = header[8]
    uop_off = header[16]
    if uop_count != 75:
        raise ValueError(f"unexpected uop_count {uop_count}")
    last_opcode = data[uop_off + 74 * 32]
    if last_opcode != UOP_END:
        raise ValueError(f"last uop is not END: opcode={last_opcode}")

    def read_uop(idx: int) -> Tuple[int, ...]:
        off = uop_off + idx * 32
        return struct.unpack("<8B4H4B4HI", data[off : off + 32])

    expected_uops = {
        5: (UOP_POOL, 0, 18, 0, 3),
        6: (UOP_POOL, 18, 8, 0, 3),
        36: (UOP_STORE, 7, 9, 0, 64),
        37: (UOP_STORE, 5, 9, 64, 64),
        38: (UOP_STORE, 8, 9, 128, 3),
        39: (UOP_AFFINE, 9, 10, 0, 131),
        70: (UOP_STORE, 14, 15, 128, 128),
    }
    for idx, expected in expected_uops.items():
        uop = read_uop(idx)
        opcode, src0, dst, c_offset, valid_c = uop[0], uop[2], uop[4], uop[16], uop[17]
        got = (opcode, src0, dst, c_offset, valid_c)
        if got != expected:
            raise ValueError(f"uop[{idx}] mismatch: got={got}, expected={expected}")


def parse_and_check_blob_v3(path: Path, audit: dict | None = None) -> None:
    data = path.read_bytes()
    if len(data) < PARAM_HEADER_BYTES:
        raise ValueError("PARAM v3 blob is shorter than header")
    header = struct.unpack("<32I", data[:PARAM_HEADER_BYTES])
    if header[0] != PARAM_BLOB_MAGIC or header[1] != PARAM_BLOB_VERSION_SCHED:
        raise ValueError(f"bad PARAM v3 magic/version: magic=0x{header[0]:08x} version={header[1]}")

    counts = {
        "tensor_desc_count": header[2],
        "scale_desc_count": header[3],
        "conv_exec_desc_count": header[4],
        "legacy_uop_count": header[8],
        "exec_entry_count": header[9],
        "window_sched_count": header[28],
        "window_pack_cmd_count": header[29],
        "row_consumer_desc_count": header[31],
    }
    if counts["tensor_desc_count"] != len(TENSOR_PLAN):
        raise ValueError(f"unexpected tensor count: {counts['tensor_desc_count']}")
    if counts["legacy_uop_count"] != 75:
        raise ValueError(f"unexpected legacy uop count: {counts['legacy_uop_count']}")
    if counts["conv_exec_desc_count"] != len(CONV_PLAN):
        raise ValueError(f"unexpected conv exec count: {counts['conv_exec_desc_count']}")

    offsets = {
        "tensor_desc": header[10],
        "scale_desc": header[11],
        "legacy_uop_debug": header[16],
        "weight_packed": header[17],
        "conv_qparam": header[18],
        "affine_qparam": header[19],
        "add_qparam": header[20],
        "pool_qparam": header[21],
        "conv_exec_desc": header[22],
        "window_sched_desc": header[23],
        "window_pack_cmd": header[24],
        "row_consumer_desc": header[25],
        "exec_plan": header[27],
    }
    for name, off in offsets.items():
        if off % SECTION_ALIGN != 0:
            raise ValueError(f"{name} offset is not {SECTION_ALIGN}B-aligned: {off}")
        if off >= len(data):
            raise ValueError(f"{name} offset out of range: {off} >= {len(data)}")
    sorted_offsets = sorted(offsets.items(), key=lambda x: x[1])
    for (name_a, off_a), (name_b, off_b) in zip(sorted_offsets, sorted_offsets[1:]):
        if off_a > off_b:
            raise ValueError(f"section order is not monotonic: {name_a}->{name_b}")

    tensor_descs: List[TensorDesc] = []
    tensor_off = offsets["tensor_desc"]
    for tid in range(counts["tensor_desc_count"]):
        off = tensor_off + tid * 16
        bank_id, _layout_tag, phys_c, base_offset, h, w, c, c_offset = struct.unpack(
            "<BBHIHHHH", data[off : off + 16]
        )
        tensor_descs.append(
            TensorDesc(
                tensor_id=tid,
                name=f"tensor_{tid}",
                bank_id=bank_id,
                base_offset=base_offset,
                h=h,
                w=w,
                c=c,
                phys_c=phys_c,
                c_offset=c_offset,
            )
        )

    exec_off = offsets["exec_plan"]
    exec_count = counts["exec_entry_count"]
    if exec_count == 0 or exec_count > MAX_EXEC_PLAN_COUNT:
        raise ValueError(f"invalid exec_entry_count={exec_count}")
    last = struct.unpack("<4B", data[exec_off + (exec_count - 1) * 4 : exec_off + exec_count * 4])
    if last[0] != EXEC_END:
        raise ValueError(f"exec_plan last entry is not EXEC_END: {last}")
    for i in range(exec_count):
        kind = data[exec_off + i * 4]
        if kind not in (EXEC_CONV, EXEC_POOL, EXEC_AFFINE, EXEC_STORE, EXEC_ADD_AFFINE, EXEC_END):
            raise ValueError(f"unsupported exec_plan[{i}] kind={kind}")

    cmd_count = counts["window_pack_cmd_count"]
    sched_count = counts["window_sched_count"]
    sched_off = offsets["window_sched_desc"]
    schedule_modes: List[int] = []
    for sid in range(sched_count):
        off = sched_off + sid * 96
        fields = struct.unpack("<4B4H", data[off : off + 12])
        mode, kernel, _stride, _dilation, in_c, k_tiles, cmd_base, cmd_len = fields
        schedule_modes.append(mode)
        if mode == WIN_MODE_INVALID or kernel not in (1, 3) or in_c == 0:
            raise ValueError(f"bad window schedule[{sid}] header: {fields}")
        if k_tiles > MAX_K_TILE_COUNT:
            raise ValueError(f"window schedule[{sid}] k_tiles={k_tiles} exceeds max")
        if cmd_base + cmd_len > cmd_count:
            raise ValueError(f"window schedule[{sid}] command range out of bounds")

    row_consumer_off = offsets["row_consumer_desc"]
    row_consumer_count = counts["row_consumer_desc_count"]
    row_consumer_modes: List[int] = []
    row_consumer_layouts: List[int] = []
    valid_store_layouts = set(STORE_LAYOUT_NAMES.keys())
    for rid in range(row_consumer_count):
        off = row_consumer_off + rid * 16
        mode = data[off]
        layout = struct.unpack("<H", data[off + 12 : off + 14])[0]
        row_consumer_modes.append(mode)
        row_consumer_layouts.append(layout)
        if mode == ROW_CONSUMER_UPSAMPLE_OUT:
            if layout != STORE_LAYOUT_NONE:
                raise ValueError(f"row_consumer[{rid}] upsample must use STORE_LAYOUT_NONE, got {layout}")
            continue
        if mode in (ROW_CONSUMER_NONE, ROW_CONSUMER_STORE, ROW_CONSUMER_ADD_STORE):
            if layout == STORE_LAYOUT_NONE:
                raise ValueError(f"row_consumer[{rid}] mode={mode} is missing scheduled store layout")
            if layout not in valid_store_layouts:
                raise ValueError(f"row_consumer[{rid}] has unknown store layout={layout}")
            if layout == STORE_LAYOUT_COLD_RMW_FALLBACK:
                raise ValueError(f"row_consumer[{rid}] selected cold RMW fallback in hot path")
        else:
            raise ValueError(f"row_consumer[{rid}] unsupported mode={mode}")

    conv_off = offsets["conv_exec_desc"]
    weight_words = (offsets["conv_qparam"] - offsets["weight_packed"]) // TK
    for cid in range(counts["conv_exec_desc_count"]):
        off = conv_off + cid * 36
        fields = struct.unpack("<4B4H4B2B2HI4H2x", data[off : off + 36])
        param_id = fields[0]
        window_sched_id = fields[2]
        row_consumer_id = fields[3]
        in_c = fields[6]
        kernel = fields[8]
        src_tensor = fields[12]
        packed_weight_word_offset = fields[16]
        weight_word_count = fields[18]
        if param_id >= len(CONV_PLAN):
            raise ValueError(f"conv_exec[{cid}] bad param_id={param_id}")
        if window_sched_id >= sched_count:
            raise ValueError(f"conv_exec[{cid}] bad window_sched_id={window_sched_id}")
        if row_consumer_id >= row_consumer_count:
            raise ValueError(f"conv_exec[{cid}] bad row_consumer_id={row_consumer_id}")
        if kernel == 1 and schedule_modes[window_sched_id] == WIN_MODE_1X1_ALIGNED:
            if src_tensor >= len(tensor_descs):
                raise ValueError(f"conv_exec[{cid}] bad src_tensor={src_tensor}")
            src_desc = tensor_descs[src_tensor]
            if in_c % TK != 0 or not tensor_desc_supports_aligned_1x1_read(src_desc):
                raise ValueError(
                    "conv_exec[{}] selects WIN_MODE_1X1_ALIGNED for non-aligned source "
                    "tensor={} phys_c={} c_offset={} base=0x{:x} in_c={}".format(
                        cid,
                        src_tensor,
                        src_desc.phys_c,
                        src_desc.c_offset,
                        src_desc.base_offset,
                        in_c,
                    )
                )
        if packed_weight_word_offset + weight_word_count > weight_words:
            raise ValueError(f"conv_exec[{cid}] packed weight range exceeds WBUF section")

    if audit is not None:
        coverage = audit.get("fusion_coverage", {})
        if coverage.get("add_fused") != coverage.get("add_total"):
            raise ValueError(f"audit ADD fusion incomplete: {coverage}")
        if not audit.get("hard_checks", {}).get("packed_weight_fits_wbuf", False):
            raise ValueError("audit reports packed weight WBUF overflow")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        default=r"D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_3ep",
        help="Existing quantized artifact directory produced from D:\\ESPNet.",
    )
    parser.add_argument(
        "--out-dir",
        default=r"D:\ESP_INT8\hw_artifacts\sched_v3_single",
        help="Hardware-facing output directory.",
    )
    parser.add_argument(
        "--param-version",
        type=int,
        choices=(1, 3),
        default=3,
        help="1 exports the legacy P6 blob; 3 exports the P7 schedule blob.",
    )
    parser.add_argument(
        "--strict-zp",
        action="store_true",
        help="Fail if non-zero activation zp or bypass scale mismatches are detected.",
    )
    parser.add_argument(
        "--strict-schedule",
        action="store_true",
        help="Fail if PARAM v3 schedule, weight packing, or fusion audit is incomplete.",
    )
    args = parser.parse_args()

    if args.param_version == 1:
        manifest = build_blob_v1(args)
        parse_and_check_blob(Path(manifest["param_blob"]))
    else:
        manifest = build_blob_v3(args)
        parse_and_check_blob_v3(Path(manifest["param_blob"]), load_json(Path(args.out_dir) / "param_audit.json"))

    print(f"Exported param blob: {manifest['param_blob']}")
    print(f"Output directory: {args.out_dir}")
    print(f"PARAM version: {manifest.get('param_version', PARAM_BLOB_VERSION)}")
    print(f"UOP count: {manifest['uop_count']}")
    if "exec_entry_count" in manifest:
        print(f"Exec entries: {manifest['exec_entry_count']}")
    print(f"Warnings: {len(manifest['warnings'])}")
    for warning in manifest["warnings"][:8]:
        print(f"  - {warning}")
    if len(manifest["warnings"]) > 8:
        print(f"  - ... {len(manifest['warnings']) - 8} more warnings in export_manifest.json")


if __name__ == "__main__":
    main()
