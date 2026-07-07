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
PARAM_BLOB_VERSION_SCHED = 4
SECTION_ALIGN = 64
PARAM_HEADER_WORDS = 32
PARAM_HEADER_BYTES = PARAM_HEADER_WORDS * 4

TM = 32
TK = 32
WBUF_BYTES = 120 * 1024
MAX_K_TILE_COUNT = 40
MAX_PACK_CMDS_PER_KT = 9
MAX_STAGED_WINDOW_PACK_CMDS = 64
MAX_WINDOW_PACK_CMD_COUNT = 256
MAX_FIXED_EXEC_DESC_COUNT = 64
MAX_EXEC_PLAN_COUNT = 96
MAX_BLOCK5_SCHED_COUNT = 8

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

PACK_CMD_VALID = 1 << 0
PACK_CMD_ZERO = 1 << 1
PACK_CMD_CONTIG_READ = 1 << 2
PACK_CMD_ALIGNED_READ = 1 << 3

ROW_CONSUMER_NONE = 0
ROW_CONSUMER_STORE = 1
ROW_CONSUMER_ADD_STORE = 2
ROW_CONSUMER_ADD_AFFINE_STORE = 3
ROW_CONSUMER_UPSAMPLE_OUT = 4
ROW_CONSUMER_AFFINE_STORE = 5
ROW_CONSUMER_CAT_AFFINE_STORE = 6
ROW_CONSUMER_ADD_STORE_BLOCK_ADD_AFFINE = 7

STORE_LAYOUT_NONE = 0
STORE_LAYOUT_COMPACT_C2 = 1
STORE_LAYOUT_COMPACT_C12 = 2
STORE_LAYOUT_COMPACT_C16 = 3
STORE_LAYOUT_COMPACT_C25 = 4
STORE_LAYOUT_COMPACT_C28 = 5
STORE_LAYOUT_C16_INTO_C19 = 6
STORE_LAYOUT_ALIGNED_TILE_COPY = 7
STORE_LAYOUT_NARROW_FIXED = 8
STORE_LAYOUT_COMPACT_C19 = 9
STORE_LAYOUT_PREFIX_ZERO_PAD = 10
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
    STORE_LAYOUT_COMPACT_C19: "COMPACT_C19",
    STORE_LAYOUT_PREFIX_ZERO_PAD: "PREFIX_ZERO_PAD",
    STORE_LAYOUT_COLD_RMW_FALLBACK: "COLD_RMW_FALLBACK",
}

ROW_CONSUMER_NAMES = {
    ROW_CONSUMER_NONE: "NONE",
    ROW_CONSUMER_STORE: "STORE",
    ROW_CONSUMER_ADD_STORE: "ADD_STORE",
    ROW_CONSUMER_ADD_AFFINE_STORE: "ADD_AFFINE_STORE",
    ROW_CONSUMER_UPSAMPLE_OUT: "UPSAMPLE_OUT",
    ROW_CONSUMER_AFFINE_STORE: "AFFINE_STORE",
    ROW_CONSUMER_CAT_AFFINE_STORE: "CAT_AFFINE_STORE",
    ROW_CONSUMER_ADD_STORE_BLOCK_ADD_AFFINE: "ADD_STORE_BLOCK_ADD_AFFINE",
}

EXEC_NOP = 0
EXEC_CONV = 1
EXEC_POOL = 2
EXEC_BLOCK_AFFINE = 6
EXEC_BLOCK_ADD_AFFINE = 7
EXEC_END = 255

FIXED_FLAG_BLOCK5_AFFINE = 1 << 0
FIXED_FLAG_BLOCK5_ADD_AFFINE = 1 << 1
FIXED_FLAG_BLOCK5_ROW_GROUP = 1 << 2
FIXED_FLAG_ROW_CONTIGUOUS_STORE = 1 << 7

BLOCK5_ADD_TENSOR_NONE = 0xFF
BLOCK5_PATTERN_INVALID = 0
BLOCK5_PATTERN_L2_C16_4C12 = 1
BLOCK5_PATTERN_L3_C28_4C25 = 2
BLOCK5_FINALIZER_INVALID = 0
BLOCK5_FINALIZER_L2 = 1
BLOCK5_FINALIZER_L3 = 2

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

FMBUF_BYTES = 0x598000
FMEM0_BYTES = 0x418000
FMBUF_POOL1_BASE = 0x3E0000
FMBUF_POOL_TMP_BASE = 0x440000
FMBUF_L20_BASE = 0x200000
FMBUF_L20_PHYS_C = 64
FMBUF_L20_C_OFFSET = 0
FMBUF_L30_BASE = 0x418000
FMBUF_L2_SCRATCH_BASE = 0x418000
FMBUF_L2_SCRATCH_SLOT_BYTES = 128 * 256 * 16
FMBUF_L30_SCRATCH_C1_BASE = 0x518000
FMBUF_L30_SCRATCH_LOW_BASE = 0x000000
FMBUF_L30_SCRATCH_SLOT_BYTES = 64 * 128 * 25
FMBUF_L3B0_SCRATCH_BASE = 0x200000
FMBUF_L3B0_SCRATCH_SLOT_BYTES = 64 * 128 * 25
BRAM_SCR0_BYTES = 256 * 512 * 3
BRAM_SCR1_BYTES = 128 * 256 * 3
FMBUF_POOL2_ALIAS_BASE = FMBUF_POOL1_BASE
FMBUF_POOL2_ALIAS_BYTES = BRAM_SCR1_BYTES


def pack_block5_reserved1(src2: int, src3: int, src4: int, add_tensor: int = BLOCK5_ADD_TENSOR_NONE) -> int:
    for name, value in (
        ("src2", src2),
        ("src3", src3),
        ("src4", src4),
        ("add_tensor", add_tensor),
    ):
        if not 0 <= int(value) <= 0xFF:
            raise ValueError(f"BLOCK5 {name} id out of u8 range: {value}")
    return int(src2) | (int(src3) << 8) | (int(src4) << 16) | (int(add_tensor) << 24)


def unpack_block5_reserved1(reserved1: int) -> Tuple[int, int, int, int]:
    raw = int(reserved1) & 0xFFFFFFFF
    return raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF, (raw >> 24) & 0xFF


def block5_source_channels(pattern: int) -> Tuple[int, int, int, int, int]:
    if pattern == BLOCK5_PATTERN_L2_C16_4C12:
        return (16, 12, 12, 12, 12)
    if pattern == BLOCK5_PATTERN_L3_C28_4C25:
        return (28, 25, 25, 25, 25)
    raise ValueError(f"unsupported BLOCK5 source pattern: {pattern}")


def build_block5_feasibility_audit() -> dict:
    """Document whether Step-4 whole-layer compact scratch is physically legal.

    The result is intentionally conservative. A false result means the exporter
    must not generate whole-layer BLOCK5 branch tensors for that stage; the HLS
    design has to use the row-level producer/consumer composer described by the
    Step 3/4 workplan instead of silently falling back to wide slice writes.
    """
    l2_available = FMBUF_BYTES - FMBUF_L2_SCRATCH_BASE
    l2_channels = block5_source_channels(BLOCK5_PATTERN_L2_C16_4C12)
    l3_channels = block5_source_channels(BLOCK5_PATTERN_L3_C28_4C25)

    def stage(name: str, h: int, w: int, channels: Tuple[int, int, int, int, int], available: int) -> dict:
        sizes = [int(h * w * c) for c in channels]
        whole = sum(sizes)
        first4 = sum(sizes[:4])
        return {
            "stage": name,
            "h": h,
            "w": w,
            "source_channels": list(channels),
            "source_bytes": sizes,
            "whole_layer_compact_bytes": whole,
            "first4_compact_bytes": first4,
            "available_scratch_bytes": available,
            "whole_layer_compact_fits": whole <= available,
            "first4_compact_fits": first4 <= available,
            "requires_row_level_composer": whole > available,
        }

    return {
        "format": "P7_STEP34_BLOCK5_FEASIBILITY",
        "reason": "whole-layer compact branch tensors are allowed only when lifetime/memory bound is proven",
        "l2_high_scratch_bytes": l2_available,
        "stages": [
            stage("L20/L2B0", 128, 256, l2_channels, l2_available),
            # Level3 has enough aggregate bytes in the current FMEM0 scratch
            # map, but it should still follow the same BLOCK5 ABI as Level2.
            stage("L30/L3B0", 64, 128, l3_channels, FMEM0_BYTES),
        ],
    }


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
    padding: int
    cache_chunks: int
    cache_col_slots: int
    flags: int
    in_c: int
    out_w: int
    k_tiles: int
    cmd_base: int
    cmd_count: int
    kt_cmd_base: Tuple[int, ...]
    reserved: Tuple[int, ...] = ()


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
class FixedExecDesc:
    kind: int
    src0_tensor: int
    src1_tensor: int
    dst_tensor: int
    param_id: int
    add_param_id: int = 0
    act_type: int = ACT_NONE
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


@dataclass(frozen=True)
class ExecPlanEntry:
    kind: int
    desc_id: int
    logical_uop_id: int
    flags: int = 0


@dataclass(frozen=True)
class Block5SchedDesc:
    pattern: int
    branch_count: int
    first_branch_conv_id: int
    first_window_sched_id: int
    first_conv_qparam_id: int
    src_tensor: int
    dst_tensor: int
    add_tensor: int
    chain_add_qparam_id0: int
    chain_add_qparam_id1: int
    chain_add_qparam_id2: int
    residual_add_qparam_id: int
    affine_param_id: int
    affine_block_count: int
    finalizer_kind: int
    scratch_region: int
    out_h: int
    out_w: int
    row_group_h: int
    valid_c: int
    reserved0: int = 0
    reserved1: int = 0
    reserved2: int = 0


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
    TensorDesc(6, "T_L2B0_CAT", BANK_FMEM0, 0x000000, 128, 256, 64, 64, 0),
    TensorDesc(7, "T_L2B0_ACT", BANK_FMEM0, 0x000000, 128, 256, 64, 64, 0),
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
    reserved = list(desc.reserved)
    if len(reserved) > 14:
        raise ValueError(f"window schedule has too many reserved words: {len(reserved)}")
    reserved += [0] * (14 - len(reserved))
    return (
        struct.pack(
            "<8B5H",
            desc.mode,
            desc.kernel,
            desc.stride,
            desc.dilation,
            desc.padding,
            desc.cache_chunks,
            desc.cache_col_slots,
            desc.flags,
            desc.in_c,
            desc.out_w,
            desc.k_tiles,
            desc.cmd_base,
            desc.cmd_count,
        )
        + struct.pack("<41H", *kt_bases)
        + struct.pack("<14H", *reserved)
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


def pack_fixed_exec_desc(desc: FixedExecDesc) -> bytes:
    return struct.pack(
        "<8B4H4B4HI",
        desc.kind,
        desc.src0_tensor,
        desc.src1_tensor,
        desc.dst_tensor,
        desc.param_id,
        desc.add_param_id,
        desc.act_type,
        desc.flags,
        desc.in_h,
        desc.in_w,
        desc.in_c,
        desc.out_c,
        desc.kernel,
        desc.stride,
        desc.dilation,
        desc.padding,
        desc.c_offset,
        desc.valid_c,
        desc.qparam_id,
        desc.reserved0,
        desc.reserved1,
    )


def pack_exec_plan_entry(entry: ExecPlanEntry) -> bytes:
    return struct.pack("<4B", entry.kind, entry.desc_id, entry.logical_uop_id, entry.flags)


def pack_block5_sched_desc(desc: Block5SchedDesc) -> bytes:
    return struct.pack(
        "<16B6HI",
        desc.pattern,
        desc.branch_count,
        desc.first_branch_conv_id,
        desc.first_window_sched_id,
        desc.first_conv_qparam_id,
        desc.src_tensor,
        desc.dst_tensor,
        desc.add_tensor,
        desc.chain_add_qparam_id0,
        desc.chain_add_qparam_id1,
        desc.chain_add_qparam_id2,
        desc.residual_add_qparam_id,
        desc.affine_param_id,
        desc.affine_block_count,
        desc.finalizer_kind,
        desc.scratch_region,
        desc.out_h,
        desc.out_w,
        desc.row_group_h,
        desc.valid_c,
        desc.reserved0,
        desc.reserved1,
        desc.reserved2,
    )


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
        phys_c, c_offset = valid_c, 0
        base_offset = FMBUF_L2_SCRATCH_BASE + idx * FMBUF_L2_SCRATCH_SLOT_BYTES
    elif 7 <= pid <= 12:
        phys_c, c_offset = valid_c, 0
        base_offset = FMBUF_L2_SCRATCH_BASE + idx * FMBUF_L2_SCRATCH_SLOT_BYTES
    elif 13 <= pid <= 18:
        phys_c, c_offset = valid_c, 0
        base_offset = (
            FMBUF_L30_SCRATCH_LOW_BASE + (idx - 1) * FMBUF_L30_SCRATCH_SLOT_BYTES
            if idx
            else FMBUF_L30_SCRATCH_C1_BASE
        )
    elif 19 <= pid <= 24:
        phys_c, c_offset = valid_c, 0
        base_offset = FMBUF_L3B0_SCRATCH_BASE + idx * FMBUF_L3B0_SCRATCH_SLOT_BYTES
    else:
        phys_c, c_offset = valid_c, 0
        base_offset = 0
    return TensorDesc(
        tensor_id=tensor_id,
        name=f"scratch_{tensor_id:02x}",
        bank_id=BANK_FMEM0,
        base_offset=base_offset,
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
        if valid_c == 19 and out_w % 32 == 0:
            return STORE_LAYOUT_COMPACT_C19
        if valid_c == 25 and out_w % 32 == 0:
            return STORE_LAYOUT_COMPACT_C25
        if valid_c == 28 and out_w % 8 == 0:
            return STORE_LAYOUT_COMPACT_C28

    if c_offset == 0 and valid_c == 16 and desc.c == 19 and desc.phys_c == 19 and out_w % 32 == 0:
        return STORE_LAYOUT_C16_INTO_C19

    start_c = desc.c_offset + c_offset
    if c_offset == 0 and 0 < valid_c < TM and desc.phys_c >= TM and start_c % TM == 0:
        return STORE_LAYOUT_PREFIX_ZERO_PAD
    if valid_c == TM and desc.phys_c % TM == 0 and start_c % TM == 0:
        return STORE_LAYOUT_ALIGNED_TILE_COPY
    raise ValueError(
        f"unsupported scheduled store layout: tensor={dst_tensor} valid_c={valid_c} "
        f"c_offset={c_offset} phys_c={desc.phys_c}; cold RMW fallback is disabled in P7"
    )


def compact_layout_for_full_tensor(valid_c: int, out_w: int) -> int | None:
    if valid_c == 2 and out_w % 16 == 0:
        return STORE_LAYOUT_COMPACT_C2
    if valid_c == 12 and out_w % 8 == 0:
        return STORE_LAYOUT_COMPACT_C12
    if valid_c == 16 and out_w % 2 == 0:
        return STORE_LAYOUT_COMPACT_C16
    if valid_c == 19 and out_w % 32 == 0:
        return STORE_LAYOUT_COMPACT_C19
    if valid_c == 25 and out_w % 32 == 0:
        return STORE_LAYOUT_COMPACT_C25
    if valid_c == 28 and out_w % 8 == 0:
        return STORE_LAYOUT_COMPACT_C28
    return None


def classify_rmw_store_layout(
    *,
    layout: int,
    desc: TensorDesc,
    c_offset: int,
    valid_c: int,
    out_w: int,
) -> Tuple[str, str, str]:
    row_base0 = desc.base_offset + desc.c_offset
    write_base0 = row_base0 + c_offset
    phys_aligned = desc.phys_c % TM == 0
    row_base_aligned = row_base0 % TM == 0
    write_base_aligned = write_base0 % TM == 0

    if layout == STORE_LAYOUT_PREFIX_ZERO_PAD:
        if not (phys_aligned and row_base_aligned):
            return (
                "R0_PREFIX_NONALIGNED",
                "prefix_zero_pad_requires_fallback_rmw_when_phys_c_or_base_is_not_32B_aligned",
                "align_tensor_phys_c_or_rewrite_to_compact_branch_scratch",
            )
        if c_offset == 0 and valid_c <= TM:
            return (
                "R2_FULL_COMPACT_TENSOR",
                "prefix_zero_pad_is_direct_full_word_write_after_zero_pad",
                "keep_prefix_zero_pad_aligned_only_or_replace_with_compact_writer",
            )
        return (
            "R4_COLD_DEBUG_ONLY",
            "unexpected_prefix_zero_pad_shape",
            "reject_from_hot_path_unless_proven_unreachable",
        )

    if layout == STORE_LAYOUT_NARROW_FIXED:
        if valid_c == TM and phys_aligned and write_base_aligned:
            return (
                "R1_FULL_WORD_ALIGNED",
                "slice_is_exactly_one_aligned_32B_tile",
                "rewrite_to_STORE_LAYOUT_ALIGNED_TILE_COPY",
            )
        compact_layout = compact_layout_for_full_tensor(valid_c, out_w)
        if (
            compact_layout is not None
            and c_offset == 0
            and desc.c == valid_c
            and desc.phys_c == desc.c
        ):
            return (
                "R2_FULL_COMPACT_TENSOR",
                f"full_tensor_can_use_{STORE_LAYOUT_NAMES[compact_layout]}",
                "rewrite_to_matching_compact_writer",
            )
        return (
            "R3_BLOCK_COMPOSER_REQUIRED",
            "slice_into_wide_concat_tensor_requires_read_modify_write",
            "rewrite_to_compact_branch_scratch_and_block_finalizer",
        )

    if layout == STORE_LAYOUT_COLD_RMW_FALLBACK:
        return (
            "R4_COLD_DEBUG_ONLY",
            "cold_rmw_fallback_selected",
            "remove_from_current_artifact_hot_path",
        )

    return ("OK", "not_a_rmw_layout", "none")


def build_rmw_audit(
    *,
    uops: Sequence[Uop],
    conv_exec_descs: Sequence[ConvExecDesc],
    row_consumers: Sequence[RowConsumerDesc],
    exec_plan: Sequence[ExecPlanEntry],
) -> dict:
    problem_layouts = {
        STORE_LAYOUT_PREFIX_ZERO_PAD,
        STORE_LAYOUT_NARROW_FIXED,
        STORE_LAYOUT_COLD_RMW_FALLBACK,
    }
    conv_logical_uop: Dict[int, int] = {
        entry.desc_id: entry.logical_uop_id
        for entry in exec_plan
        if entry.kind == EXEC_CONV and entry.desc_id < len(conv_exec_descs)
    }

    entries: List[dict] = []
    for conv_desc_id, conv in enumerate(conv_exec_descs):
        if conv.row_consumer_id >= len(row_consumers):
            raise ValueError(f"conv_exec[{conv_desc_id}] has bad row_consumer_id={conv.row_consumer_id}")
        consumer = row_consumers[conv.row_consumer_id]
        layout = consumer.reserved0
        if layout not in problem_layouts:
            continue

        logical_uop = conv_logical_uop.get(conv_desc_id, -1)
        if logical_uop < 0 or logical_uop >= len(uops):
            raise ValueError(f"conv_exec[{conv_desc_id}] missing logical uop mapping")
        producer = uops[logical_uop]

        if consumer.mode == ROW_CONSUMER_NONE:
            dst_tensor = conv.dst_tensor
            c_offset = conv.dst_c_offset
            valid_c = conv.valid_c
        else:
            dst_tensor = consumer.store_dst_tensor
            c_offset = consumer.store_c_offset
            valid_c = consumer.valid_c

        if dst_tensor == TID_INVALID:
            raise ValueError(f"conv_exec[{conv_desc_id}] selected RMW layout for invalid dst tensor")
        desc = resolve_compile_time_store_desc(dst_tensor, producer, valid_c)
        out_h = conv_out_dim(producer.in_h, producer.stride)
        out_w = conv_out_dim(producer.in_w, producer.stride)
        rmw_class, reason, action = classify_rmw_store_layout(
            layout=layout,
            desc=desc,
            c_offset=c_offset,
            valid_c=valid_c,
            out_w=out_w,
        )
        row_base0 = desc.base_offset + desc.c_offset
        write_base0 = row_base0 + c_offset
        entries.append(
            {
                "logical_uop": logical_uop,
                "conv_desc_id": conv_desc_id,
                "conv_param": conv.param_id,
                "layer": CONVS[conv.param_id].name if 0 <= conv.param_id < len(CONVS) else f"conv_{conv.param_id}",
                "mode": ROW_CONSUMER_NAMES.get(consumer.mode, str(consumer.mode)),
                "layout": STORE_LAYOUT_NAMES.get(layout, str(layout)),
                "dst_tensor": dst_tensor,
                "dst_tensor_name": desc.name,
                "dst_c": desc.c,
                "dst_phys_c": desc.phys_c,
                "dst_base": f"0x{desc.base_offset:06x}",
                "dst_c_offset": desc.c_offset,
                "store_c_offset": c_offset,
                "valid_c": valid_c,
                "out_h": out_h,
                "out_w": out_w,
                "row_base0": f"0x{row_base0:06x}",
                "write_base0": f"0x{write_base0:06x}",
                "phys_c_aligned_32B": desc.phys_c % TM == 0,
                "row_base_aligned_32B": row_base0 % TM == 0,
                "write_base_aligned_32B": write_base0 % TM == 0,
                "reason": reason,
                "class": rmw_class,
                "action": action,
            }
        )

    class_counts: Dict[str, int] = {}
    layout_counts: Dict[str, int] = {}
    for entry in entries:
        class_counts[entry["class"]] = class_counts.get(entry["class"], 0) + 1
        layout_counts[entry["layout"]] = layout_counts.get(entry["layout"], 0) + 1

    return {
        "format": "ESP_INT8_P7F_RMW_AUDIT_V1",
        "description": "Rows requiring PREFIX/NARROW/COLD_RMW handling before P7F hot-path no-RMW synthesis.",
        "problem_count": len(entries),
        "layout_counts": layout_counts,
        "class_counts": class_counts,
        "entries": entries,
    }


def window_mode_for(in_c: int, kernel: int, aligned_1x1_source: bool = True) -> int:
    if kernel == 1:
        return WIN_MODE_1X1_ALIGNED if in_c % TK == 0 and aligned_1x1_source else WIN_MODE_1X1_PACKED
    if kernel != 3:
        raise ValueError(f"unsupported kernel size: {kernel}")
    staged_modes = {
        3: WIN_MODE_3X3_STAGED_C3,
        12: WIN_MODE_3X3_STAGED_C12,
        19: WIN_MODE_3X3_STAGED_C19,
        25: WIN_MODE_3X3_STAGED_C25,
        28: WIN_MODE_3X3_STAGED_C28,
        64: WIN_MODE_3X3_STAGED_C64,
        128: WIN_MODE_3X3_STAGED_C128,
        131: WIN_MODE_3X3_STAGED_C131,
    }
    if in_c in staged_modes:
        return staged_modes[in_c]
    raise ValueError(f"unsupported P7 staged 3x3 input channel count: in_c={in_c}")


def window_cache_chunks_for(in_c: int, kernel: int) -> int:
    if kernel != 3:
        return 0
    chunks = math.ceil(in_c / TK)
    if chunks <= 0 or chunks > 5:
        raise ValueError(f"unsupported staged 3x3 chunk count for in_c={in_c}: {chunks}")
    return chunks


def make_window_pack_schedule(
    in_c: int,
    kernel: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
    out_w: int = 0,
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
            # The staged HLS WinGen stores source activations in 32-byte channel
            # chunks. A single pack command is intentionally chunk-local; if it
            # crossed c31/c32, HLS would read only the first cache word.
            chunk_remaining = tk - (cin % tk)
            count = min(kt_end - k, in_c - cin, tk - dst_lane, chunk_remaining)
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
        stride=stride,
        dilation=dilation,
        padding=padding,
        cache_chunks=window_cache_chunks_for(in_c, kernel),
        cache_col_slots=3 if kernel == 3 else 0,
        flags=0,
        in_c=in_c,
        out_w=out_w,
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
    cache: Dict[Tuple[int, int, int, int, int, int, int], int] = {}

    for uop in uops:
        if uop.opcode != UOP_CONV:
            continue
        source_aligned = tensor_supports_aligned_1x1_read(uop.src0)
        mode = window_mode_for(uop.in_c, uop.kernel, source_aligned)
        out_w = conv_out_dim(uop.in_w, uop.stride)
        key = (uop.kernel, uop.in_c, mode, uop.stride, uop.dilation, uop.padding, out_w)
        if key not in cache:
            local_desc, _legacy_cmds = make_window_pack_schedule(
                uop.in_c,
                uop.kernel,
                stride=uop.stride,
                dilation=uop.dilation,
                padding=uop.padding,
                out_w=out_w,
                mode=mode,
            )
            global_desc = WindowSchedDesc(
                mode=local_desc.mode,
                kernel=local_desc.kernel,
                stride=local_desc.stride,
                dilation=local_desc.dilation,
                padding=local_desc.padding,
                cache_chunks=local_desc.cache_chunks,
                cache_col_slots=local_desc.cache_col_slots,
                flags=local_desc.flags,
                in_c=local_desc.in_c,
                out_w=local_desc.out_w,
                k_tiles=local_desc.k_tiles,
                cmd_base=0,
                cmd_count=0,
                kt_cmd_base=[0] * (MAX_K_TILE_COUNT + 1),
            )
            cache[key] = len(schedules)
            schedules.append(global_desc)
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
                "shape_key": [
                    desc.kernel,
                    desc.in_c,
                    desc.mode,
                    desc.stride,
                    desc.dilation,
                    desc.padding,
                    desc.out_w,
                ],
                "cache_chunks": desc.cache_chunks,
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


def _build_block_add_affine_redirects(uops: Sequence[Uop]) -> Dict[int, Tuple[int, Uop, Uop]]:
    redirects: Dict[int, Tuple[int, Uop, Uop]] = {}
    for idx, uop in enumerate(uops):
        if (
            uop.opcode == UOP_ADD
            and idx + 1 < len(uops)
            and uops[idx + 1].opcode == UOP_AFFINE
            and uops[idx + 1].src0 == uop.dst
            and uop.dst in (6, 13)
        ):
            redirects[uop.dst] = (idx, uop, uops[idx + 1])
    return redirects


def _build_block_affine_redirects(uops: Sequence[Uop]) -> Dict[int, Tuple[int, Uop]]:
    redirects: Dict[int, Tuple[int, Uop]] = {}
    for idx, uop in enumerate(uops):
        if uop.opcode == UOP_AFFINE and uop.src0 in (4, 11):
            redirects[uop.src0] = (idx, uop)
    return redirects


def build_exec_plan_sections(
    uops: Sequence[Uop],
    schedule_ids_by_param: Dict[int, int],
    packed_weight_word_offsets: Dict[int, int],
    weight_report: Sequence[dict],
) -> Tuple[List[ConvExecDesc], List[RowConsumerDesc], List[FixedExecDesc], List[ExecPlanEntry], List[Block5SchedDesc], dict]:
    weight_words_by_param = {int(x["param_id"]): int(x["packed_words"]) for x in weight_report}
    k_tiles_by_param = {int(x["param_id"]): int(x["k_tiles"]) for x in weight_report}
    conv_exec_descs: List[ConvExecDesc] = []
    row_consumers: List[RowConsumerDesc] = []
    fixed_exec_descs: List[FixedExecDesc] = []
    exec_plan: List[ExecPlanEntry] = []
    block5_sched_descs: List[Block5SchedDesc] = []
    fusion_audit: List[dict] = []
    skipped: set[int] = set()

    add_total = sum(1 for u in uops if u.opcode == UOP_ADD)
    store_total = sum(1 for u in uops if u.opcode == UOP_STORE)
    affine_total = sum(1 for u in uops if u.opcode == UOP_AFFINE)
    add_fused = 0
    store_as_row_consumer = 0
    standalone_store_fixed = 0
    affine_fused = 0
    standalone_affine = 0
    block_add_affine_redirects = _build_block_add_affine_redirects(uops)
    block_affine_redirects = _build_block_affine_redirects(uops)
    fused_affine_indices: set[int] = set()

    def mark_affine_fused(affine_idx: int) -> None:
        nonlocal affine_fused
        if affine_idx not in fused_affine_indices:
            fused_affine_indices.add(affine_idx)
            skipped.add(affine_idx)
            affine_fused += 1

    def append_conv_entry(uop: Uop, row_consumer: RowConsumerDesc, *, dst_tensor: int | None = None) -> int:
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
                dst_tensor=uop.dst if dst_tensor is None else dst_tensor,
                dst_c_offset=uop.c_offset,
                valid_c=uop.valid_c,
                packed_weight_word_offset=packed_weight_word_offsets[uop.param_id],
                k_tiles=k_tiles_by_param[uop.param_id],
                weight_words=weight_words_by_param[uop.param_id],
                flags=uop.flags,
            )
        )
        exec_plan.append(ExecPlanEntry(EXEC_CONV, conv_id, idx, 0))
        return conv_id

    def append_conv_desc_only(uop: Uop) -> int:
        row_id = len(row_consumers)
        row_consumers.append(
            RowConsumerDesc(
                mode=ROW_CONSUMER_NONE,
                store_dst_tensor=TID_INVALID,
                store_c_offset=0,
                valid_c=uop.valid_c,
                reserved0=STORE_LAYOUT_NONE,
            )
        )
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
                dst_tensor=TID_INVALID,
                dst_c_offset=0,
                valid_c=uop.valid_c,
                packed_weight_word_offset=packed_weight_word_offsets[uop.param_id],
                k_tiles=k_tiles_by_param[uop.param_id],
                weight_words=weight_words_by_param[uop.param_id],
                flags=uop.flags,
            )
        )
        return conv_id

    block5_group_specs = {
        8: {
            "name": "L20",
            "pattern": BLOCK5_PATTERN_L2_C16_4C12,
            "branches": [8, 9, 11, 14, 17],
            "chain_adds": [12, 15, 18],
            "stores": [10, 13, 16, 19],
            "affine": 20,
            "residual_add": None,
            "residual_tensor": TID_INVALID,
        },
        22: {
            "name": "L2B0",
            "pattern": BLOCK5_PATTERN_L2_C16_4C12,
            "branches": [22, 23, 25, 28, 31],
            "chain_adds": [26, 29, 32],
            "stores": [24, 27, 30, 33],
            "affine": 35,
            "residual_add": 34,
            "residual_tensor": 5,
        },
        41: {
            "name": "L30",
            "pattern": BLOCK5_PATTERN_L3_C28_4C25,
            "branches": [41, 42, 44, 47, 50],
            "chain_adds": [45, 48, 51],
            "stores": [43, 46, 49, 52],
            "affine": 53,
            "residual_add": None,
            "residual_tensor": TID_INVALID,
        },
        55: {
            "name": "L3B0",
            "pattern": BLOCK5_PATTERN_L3_C28_4C25,
            "branches": [55, 56, 58, 61, 64],
            "chain_adds": [59, 62, 65],
            "stores": [57, 60, 63, 66],
            "affine": 68,
            "residual_add": 67,
            "residual_tensor": 12,
        },
    }

    def append_block5_row_group(start_idx: int, spec: dict) -> None:
        nonlocal add_fused, store_as_row_consumer, affine_fused
        branches = [uops[i] for i in spec["branches"]]
        affine = uops[spec["affine"]]
        chain_adds = [uops[i] for i in spec["chain_adds"]]
        stores = [uops[i] for i in spec["stores"]]
        residual_idx = spec["residual_add"]
        residual_add = uops[residual_idx] if residual_idx is not None else None
        channels = block5_source_channels(spec["pattern"])

        if any(b.opcode != UOP_CONV for b in branches):
            raise ValueError(f"{spec['name']} BLOCK5 branches are not all CONV")
        if [b.out_c for b in branches] != list(channels):
            raise ValueError(f"{spec['name']} BLOCK5 channel order mismatch: {[b.out_c for b in branches]}")
        if len({b.src0 for b in branches}) != 1:
            raise ValueError(f"{spec['name']} BLOCK5 branches must share one compact source")
        if any(a.opcode != UOP_ADD for a in chain_adds):
            raise ValueError(f"{spec['name']} BLOCK5 chain adds are not all ADD")
        if [a.param_id for a in chain_adds] != list(range(chain_adds[0].param_id, chain_adds[0].param_id + 3)):
            raise ValueError(f"{spec['name']} BLOCK5 chain add qparams must be contiguous")
        if any(s.opcode != UOP_STORE for s in stores):
            raise ValueError(f"{spec['name']} BLOCK5 skipped stores are not all STORE")
        if affine.opcode != UOP_AFFINE:
            raise ValueError(f"{spec['name']} BLOCK5 final op is not AFFINE")
        if residual_add is not None and residual_add.opcode != UOP_ADD:
            raise ValueError(f"{spec['name']} BLOCK5 residual op is not ADD")

        base_conv_desc_id = len(conv_exec_descs)
        branch_desc_ids = [append_conv_desc_only(b) for b in branches]
        if branch_desc_ids != list(range(base_conv_desc_id, base_conv_desc_id + 5)):
            raise ValueError(f"{spec['name']} BLOCK5 branch desc ids are not contiguous")
        if [schedule_ids_by_param[b.param_id] for b in branches] != list(
            range(schedule_ids_by_param[branches[0].param_id], schedule_ids_by_param[branches[0].param_id] + 5)
        ):
            raise ValueError(f"{spec['name']} BLOCK5 window sched ids are not contiguous")
        if [b.qparam_id if b.qparam_id else b.param_id for b in branches] != list(
            range(branches[0].qparam_id if branches[0].qparam_id else branches[0].param_id,
                  (branches[0].qparam_id if branches[0].qparam_id else branches[0].param_id) + 5)
        ):
            raise ValueError(f"{spec['name']} BLOCK5 conv qparams must be contiguous")

        flags = affine.flags | FIXED_FLAG_BLOCK5_ROW_GROUP
        kind = EXEC_BLOCK_AFFINE
        add_param_id = 0
        add_tensor = BLOCK5_ADD_TENSOR_NONE
        if residual_add is not None:
            kind = EXEC_BLOCK_ADD_AFFINE
            flags |= FIXED_FLAG_BLOCK5_ADD_AFFINE
            add_param_id = residual_add.param_id
            add_tensor = spec["residual_tensor"]
        else:
            flags |= FIXED_FLAG_BLOCK5_AFFINE

        block5_sched_id = len(block5_sched_descs)
        if block5_sched_id >= MAX_BLOCK5_SCHED_COUNT:
            raise ValueError(f"BLOCK5 schedule count exceeds MAX_BLOCK5_SCHED_COUNT={MAX_BLOCK5_SCHED_COUNT}")
        branch0_qparam = branches[0].qparam_id if branches[0].qparam_id else branches[0].param_id
        out_h = branches[0].in_h // max(1, branches[0].stride)
        out_w = branches[0].in_w // max(1, branches[0].stride)
        block5_sched_descs.append(
            Block5SchedDesc(
                pattern=spec["pattern"],
                branch_count=5,
                first_branch_conv_id=base_conv_desc_id,
                first_window_sched_id=schedule_ids_by_param[branches[0].param_id],
                first_conv_qparam_id=branch0_qparam,
                src_tensor=branches[0].src0,
                dst_tensor=affine.dst,
                add_tensor=add_tensor,
                chain_add_qparam_id0=chain_adds[0].param_id,
                chain_add_qparam_id1=chain_adds[1].param_id,
                chain_add_qparam_id2=chain_adds[2].param_id,
                residual_add_qparam_id=add_param_id,
                affine_param_id=affine.param_id,
                affine_block_count=4 if spec["pattern"] == BLOCK5_PATTERN_L3_C28_4C25 else 2,
                finalizer_kind=(
                    BLOCK5_FINALIZER_L3
                    if spec["pattern"] == BLOCK5_PATTERN_L3_C28_4C25
                    else BLOCK5_FINALIZER_L2
                ),
                scratch_region=spec["pattern"],
                out_h=out_h,
                out_w=out_w,
                row_group_h=64,
                valid_c=affine.valid_c,
            )
        )

        fixed_id = len(fixed_exec_descs)
        fixed_exec_descs.append(
            FixedExecDesc(
                kind=kind,
                src0_tensor=branches[0].src0,
                src1_tensor=TID_INVALID,
                dst_tensor=affine.dst,
                param_id=affine.param_id,
                add_param_id=add_param_id,
                act_type=affine.act_type,
                flags=flags,
                in_h=branches[0].in_h,
                in_w=branches[0].in_w,
                in_c=branches[0].in_c,
                out_c=affine.out_c,
                c_offset=base_conv_desc_id,
                valid_c=affine.valid_c,
                qparam_id=chain_adds[0].param_id,
                reserved0=spec["pattern"] | (block5_sched_id << 8),
                reserved1=pack_block5_reserved1(TID_INVALID, TID_INVALID, TID_INVALID, add_tensor),
            )
        )
        exec_plan.append(ExecPlanEntry(kind, fixed_id, spec["affine"], flags))

        skipped.update(spec["branches"])
        skipped.update(spec["chain_adds"])
        skipped.update(spec["stores"])
        skipped.add(spec["affine"])
        if residual_idx is not None:
            skipped.add(residual_idx)

        add_fused += len(spec["chain_adds"]) + (1 if residual_idx is not None else 0)
        store_as_row_consumer += len(spec["stores"])
        affine_fused += 1
        fusion_audit.append(
            {
                "logical_uop": start_idx,
                "kind": "BLOCK5_ROW_GROUP_ADD_AFFINE" if residual_idx is not None else "BLOCK5_ROW_GROUP_AFFINE",
                "name": spec["name"],
                "fixed_desc": fixed_id,
                "block5_sched_id": block5_sched_id,
                "branch_conv_desc_base": base_conv_desc_id,
                "branch_window_sched_base": schedule_ids_by_param[branches[0].param_id],
                "branch_qparam_base": branch0_qparam,
                "branch_uops": list(spec["branches"]),
                "chain_add_uops": list(spec["chain_adds"]),
                "skipped_store_uops": list(spec["stores"]),
                "residual_add_uop": residual_idx,
                "affine_uop": spec["affine"],
                "source_channels": list(channels),
                "src_tensor": branches[0].src0,
                "dst": affine.dst,
                "pattern": spec["pattern"],
                "row_consumer": "BLOCK5_ROW_GROUP",
                "store_layout": "ALIGNED_FULL_TILE_FINAL",
            }
        )

    def apply_block_add_affine_redirect_if_safe(
        consumer: RowConsumerDesc,
        *,
        conv: Uop,
        store_dst_tensor: int,
        c_offset: int,
        valid_c: int,
        audit: dict,
    ) -> Tuple[RowConsumerDesc, dict]:
        nonlocal add_fused
        redirect = block_add_affine_redirects.get(store_dst_tensor)
        if redirect is None or consumer.mode != ROW_CONSUMER_ADD_STORE:
            return consumer, audit
        add_idx, add, affine = redirect
        tensor = TENSOR_DESC_BY_ID.get(store_dst_tensor)
        if tensor is None:
            return consumer, audit
        if c_offset + valid_c != tensor.c:
            return consumer, audit
        residual = add.src1 if add.src0 == store_dst_tensor else add.src0
        skipped.update({add_idx, add_idx + 1})
        add_fused += 1
        mark_affine_fused(add_idx + 1)
        redirected = RowConsumerDesc(
            mode=ROW_CONSUMER_ADD_STORE_BLOCK_ADD_AFFINE,
            add_other_tensor=consumer.add_other_tensor,
            store_dst_tensor=consumer.store_dst_tensor,
            add_qparam_id=consumer.add_qparam_id,
            store_c_offset=consumer.store_c_offset,
            valid_c=consumer.valid_c,
            alias_tensor=affine.dst,
            affine_param_id=affine.param_id,
            affine_block_base=add.param_id,
            act_type=affine.act_type,
            reserved0=consumer.reserved0,
            reserved1=residual,
        )
        audit = {
            **audit,
            "kind": f"{audit.get('kind', 'CONV')}_BLOCK_ADD_AFFINE",
            "block_add_uop": add_idx,
            "block_affine_uop": add_idx + 1,
            "block_residual_tensor": residual,
            "block_final_tensor": affine.dst,
            "row_consumer": "ADD_STORE_BLOCK_ADD_AFFINE",
            "store_layout": STORE_LAYOUT_NAMES.get(consumer.reserved0, str(consumer.reserved0)),
        }
        return redirected, audit

    def apply_block_affine_redirect_if_safe(
        consumer: RowConsumerDesc,
        *,
        store_dst_tensor: int,
        c_offset: int,
        valid_c: int,
        audit: dict,
    ) -> Tuple[RowConsumerDesc, dict]:
        redirect = block_affine_redirects.get(store_dst_tensor)
        if redirect is None:
            return consumer, audit
        if consumer.mode not in (ROW_CONSUMER_STORE, ROW_CONSUMER_ADD_STORE):
            return consumer, audit
        tensor = TENSOR_DESC_BY_ID.get(store_dst_tensor)
        if tensor is None or c_offset + valid_c != tensor.c:
            return consumer, audit
        affine_idx, affine = redirect
        mark_affine_fused(affine_idx)
        redirected = RowConsumerDesc(
            mode=ROW_CONSUMER_ADD_STORE_BLOCK_ADD_AFFINE,
            add_other_tensor=consumer.add_other_tensor,
            store_dst_tensor=consumer.store_dst_tensor,
            add_qparam_id=consumer.add_qparam_id,
            store_c_offset=consumer.store_c_offset,
            valid_c=consumer.valid_c,
            alias_tensor=affine.dst,
            affine_param_id=affine.param_id,
            affine_block_base=0,
            act_type=affine.act_type,
            reserved0=consumer.reserved0,
            reserved1=TID_INVALID,
        )
        audit = {
            **audit,
            "kind": f"{audit.get('kind', 'CONV')}_BLOCK_AFFINE",
            "block_affine_uop": affine_idx,
            "block_final_tensor": affine.dst,
            "row_consumer": "ADD_STORE_BLOCK_AFFINE",
            "store_layout": STORE_LAYOUT_NAMES.get(consumer.reserved0, str(consumer.reserved0)),
        }
        return redirected, audit

    for idx, uop in enumerate(uops):
        if idx in skipped or uop.opcode in (UOP_LOAD_FM, UOP_NOP):
            continue
        if uop.opcode == UOP_END:
            break

        if idx in block5_group_specs:
            append_block5_row_group(idx, block5_group_specs[idx])
            continue

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
            if (
                idx + 2 < len(uops)
                and uops[idx + 1].opcode == UOP_STORE
                and uops[idx + 1].dst == uop.dst
                and uops[idx + 2].opcode == UOP_AFFINE
                and uops[idx + 2].src0 == uop.dst
                and uop.out_c + uops[idx + 1].valid_c == uops[idx + 2].valid_c
                and uops[idx + 1].c_offset == uop.out_c
            ):
                store = uops[idx + 1]
                affine = uops[idx + 2]
                layout = store_layout_for(
                    dst_tensor=affine.dst,
                    producer=uop,
                    c_offset=0,
                    valid_c=affine.valid_c,
                    out_w=out_w,
                )
                row_consumer = RowConsumerDesc(
                    mode=ROW_CONSUMER_CAT_AFFINE_STORE,
                    add_other_tensor=store.src0,
                    store_dst_tensor=affine.dst,
                    store_c_offset=0,
                    valid_c=affine.valid_c,
                    affine_param_id=affine.param_id,
                    affine_block_base=0,
                    act_type=affine.act_type,
                    reserved0=layout,
                    reserved1=0,
                )
                skipped.update({idx + 1, idx + 2})
                mark_affine_fused(idx + 2)
                store_as_row_consumer += 1
                fusion_audit.append(
                    {
                        "logical_uop": idx,
                        "kind": "CONV_CAT_AFFINE_STORE",
                        "conv_param": uop.param_id,
                        "cat_store_uop": idx + 1,
                        "affine_uop": idx + 2,
                        "row_consumer": "CAT_AFFINE_STORE",
                        "store_layout": STORE_LAYOUT_NAMES.get(layout, str(layout)),
                    }
                )
            elif (
                idx + 3 < len(uops)
                and _uop_is_add_of(uops[idx + 1], uop.dst)
                and uops[idx + 2].opcode == UOP_AFFINE
                and uops[idx + 2].src0 == uops[idx + 1].dst
                and _uop_is_store_of(uops[idx + 3], uops[idx + 2].dst)
            ):
                add = uops[idx + 1]
                affine = uops[idx + 2]
                store = uops[idx + 3]
                other = add.src1 if add.src0 == uop.dst else add.src0
                layout = store_layout_for(
                    dst_tensor=store.dst,
                    producer=uop,
                    c_offset=store.c_offset,
                    valid_c=store.valid_c,
                    out_w=out_w,
                )
                row_consumer = RowConsumerDesc(
                    mode=ROW_CONSUMER_ADD_AFFINE_STORE,
                    add_other_tensor=other,
                    store_dst_tensor=store.dst,
                    add_qparam_id=add.param_id,
                    store_c_offset=store.c_offset,
                    valid_c=store.valid_c,
                    alias_tensor=(affine.dst if affine.dst != store.dst else TID_INVALID),
                    affine_param_id=affine.param_id,
                    affine_block_base=0,
                    act_type=affine.act_type,
                    reserved0=layout,
                )
                skipped.update({idx + 1, idx + 2, idx + 3})
                add_fused += 1
                affine_fused += 1
                store_as_row_consumer += 1
                fusion_audit.append(
                    {
                        "logical_uop": idx,
                        "kind": "CONV_ADD_AFFINE_STORE",
                        "conv_param": uop.param_id,
                        "add_uop": idx + 1,
                        "affine_uop": idx + 2,
                        "store_uop": idx + 3,
                        "row_consumer": "ADD_AFFINE_STORE",
                        "store_layout": STORE_LAYOUT_NAMES.get(layout, str(layout)),
                    }
                )
            elif (
                idx + 2 < len(uops)
                and uops[idx + 1].opcode == UOP_AFFINE
                and uops[idx + 1].src0 == uop.dst
                and _uop_is_store_of(uops[idx + 2], uops[idx + 1].dst)
            ):
                affine = uops[idx + 1]
                store = uops[idx + 2]
                layout = store_layout_for(
                    dst_tensor=store.dst,
                    producer=uop,
                    c_offset=store.c_offset,
                    valid_c=store.valid_c,
                    out_w=out_w,
                )
                row_consumer = RowConsumerDesc(
                    mode=ROW_CONSUMER_AFFINE_STORE,
                    store_dst_tensor=store.dst,
                    store_c_offset=store.c_offset,
                    valid_c=store.valid_c,
                    alias_tensor=(affine.dst if affine.dst != store.dst else TID_INVALID),
                    affine_param_id=affine.param_id,
                    affine_block_base=0,
                    act_type=affine.act_type,
                    reserved0=layout,
                )
                skipped.update({idx + 1, idx + 2})
                affine_fused += 1
                store_as_row_consumer += 1
                fusion_audit.append(
                    {
                        "logical_uop": idx,
                        "kind": "CONV_AFFINE_STORE",
                        "conv_param": uop.param_id,
                        "affine_uop": idx + 1,
                        "store_uop": idx + 2,
                        "row_consumer": "AFFINE_STORE",
                        "store_layout": STORE_LAYOUT_NAMES.get(layout, str(layout)),
                    }
                )
            elif idx + 1 < len(uops) and _uop_is_store_of(uops[idx + 1], uop.dst):
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
                fusion_audit.append(
                    {
                        "logical_uop": idx,
                        "kind": "CONV_STORE",
                        "conv_param": uop.param_id,
                        "store_uop": idx + 1,
                        "row_consumer": "UPSAMPLE_OUT" if mode == ROW_CONSUMER_UPSAMPLE_OUT else "STORE",
                        "store_layout": STORE_LAYOUT_NAMES.get(layout, str(layout)),
                    }
                )
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
                fusion_audit.append(
                    {
                        "logical_uop": idx,
                        "kind": "CONV_ADD_STORE",
                        "conv_param": uop.param_id,
                        "add_uop": idx + 1,
                        "store_uop": idx + 2,
                        "row_consumer": "ADD_STORE",
                        "store_layout": STORE_LAYOUT_NAMES.get(row_consumer.reserved0, str(row_consumer.reserved0)),
                    }
                )
            else:
                fusion_audit.append(
                    {
                        "logical_uop": idx,
                        "kind": "CONV_ROW_CONSUMER_DEFAULT",
                        "conv_param": uop.param_id,
                        "row_consumer": "NONE",
                        "store_layout": STORE_LAYOUT_NAMES.get(row_consumer.reserved0, str(row_consumer.reserved0)),
                    }
                )

            if fusion_audit:
                last_audit = fusion_audit.pop()
                row_consumer, last_audit = apply_block_add_affine_redirect_if_safe(
                    row_consumer,
                    conv=uop,
                    store_dst_tensor=row_consumer.store_dst_tensor,
                    c_offset=row_consumer.store_c_offset,
                    valid_c=row_consumer.valid_c,
                    audit=last_audit,
                )
                row_consumer, last_audit = apply_block_affine_redirect_if_safe(
                    row_consumer,
                    store_dst_tensor=row_consumer.store_dst_tensor,
                    c_offset=row_consumer.store_c_offset,
                    valid_c=row_consumer.valid_c,
                    audit=last_audit,
                )
                fusion_audit.append(last_audit)

            append_conv_entry(uop, row_consumer)
            continue

        if uop.opcode == UOP_POOL:
            fixed_id = len(fixed_exec_descs)
            fixed_exec_descs.append(
                FixedExecDesc(
                    kind=EXEC_POOL,
                    src0_tensor=uop.src0,
                    src1_tensor=TID_INVALID,
                    dst_tensor=uop.dst,
                    param_id=uop.param_id,
                    act_type=uop.act_type,
                    flags=uop.flags,
                    in_h=uop.in_h,
                    in_w=uop.in_w,
                    in_c=uop.in_c,
                    out_c=uop.out_c,
                    kernel=uop.kernel,
                    stride=uop.stride,
                    dilation=uop.dilation,
                    padding=uop.padding,
                    c_offset=uop.c_offset,
                    valid_c=uop.valid_c,
                    qparam_id=uop.qparam_id,
                )
            )
            exec_plan.append(ExecPlanEntry(EXEC_POOL, fixed_id, idx, uop.flags))
            fusion_audit.append({"logical_uop": idx, "kind": "STANDALONE_POOL", "fixed_desc": fixed_id})
            continue

        if uop.opcode == UOP_AFFINE:
            standalone_affine += 1
            raise ValueError(f"unfused AFFINE uop[{idx}] would require a removed P6 fixed-op path")

        if uop.opcode == UOP_STORE:
            if (
                idx + 3 < len(uops)
                and uops[idx + 1].opcode == UOP_STORE
                and uops[idx + 2].opcode == UOP_STORE
                and uops[idx + 3].opcode == UOP_AFFINE
                and uops[idx].dst == uops[idx + 1].dst == uops[idx + 2].dst == uops[idx + 3].src0
            ):
                affine = uops[idx + 3]
                flags = affine.flags
                if affine.valid_c == 131:
                    flags |= FIXED_FLAG_ROW_CONTIGUOUS_STORE
                fixed_id = len(fixed_exec_descs)
                fixed_exec_descs.append(
                    FixedExecDesc(
                        kind=EXEC_BLOCK_AFFINE,
                        src0_tensor=uop.src0,
                        src1_tensor=uops[idx + 1].src0,
                        dst_tensor=affine.dst,
                        param_id=affine.param_id,
                        act_type=affine.act_type,
                        flags=flags,
                        in_h=affine.in_h,
                        in_w=affine.in_w,
                        in_c=uop.valid_c,
                        out_c=affine.out_c,
                        valid_c=affine.valid_c,
                        reserved0=uops[idx + 2].src0,
                    )
                )
                exec_plan.append(ExecPlanEntry(EXEC_BLOCK_AFFINE, fixed_id, idx + 3, flags))
                skipped.update({idx + 1, idx + 2, idx + 3})
                affine_fused += 1
                fusion_audit.append(
                    {
                        "logical_uop": idx,
                        "kind": "BLOCK_AFFINE_3SRC",
                        "fixed_desc": fixed_id,
                        "store_uops": [idx, idx + 1, idx + 2],
                        "affine_uop": idx + 3,
                        "src": [uop.src0, uops[idx + 1].src0, uops[idx + 2].src0],
                        "dst": affine.dst,
                        "shape": [affine.in_h, affine.in_w, affine.valid_c],
                    }
                )
                continue
            if (
                idx + 2 < len(uops)
                and uops[idx + 1].opcode == UOP_STORE
                and uops[idx + 2].opcode == UOP_AFFINE
                and uops[idx].dst == uops[idx + 1].dst == uops[idx + 2].src0
            ):
                affine = uops[idx + 2]
                fixed_id = len(fixed_exec_descs)
                fixed_exec_descs.append(
                    FixedExecDesc(
                        kind=EXEC_BLOCK_AFFINE,
                        src0_tensor=uop.src0,
                        src1_tensor=uops[idx + 1].src0,
                        dst_tensor=affine.dst,
                        param_id=affine.param_id,
                        act_type=affine.act_type,
                        flags=affine.flags,
                        in_h=affine.in_h,
                        in_w=affine.in_w,
                        in_c=uop.valid_c,
                        out_c=affine.out_c,
                        valid_c=affine.valid_c,
                        reserved0=TID_INVALID,
                    )
                )
                exec_plan.append(ExecPlanEntry(EXEC_BLOCK_AFFINE, fixed_id, idx + 2, affine.flags))
                skipped.update({idx + 1, idx + 2})
                affine_fused += 1
                fusion_audit.append(
                    {
                        "logical_uop": idx,
                        "kind": "BLOCK_AFFINE_2SRC",
                        "fixed_desc": fixed_id,
                        "store_uops": [idx, idx + 1],
                        "affine_uop": idx + 2,
                        "src": [uop.src0, uops[idx + 1].src0],
                        "dst": affine.dst,
                        "shape": [affine.in_h, affine.in_w, affine.valid_c],
                    }
                )
                continue
            standalone_store_fixed += 1
            raise ValueError(f"unfused STORE uop[{idx}] would require a removed P6 fixed-op path")

        if uop.opcode == UOP_ADD:
            raise ValueError(f"unfused ADD uop[{idx}] is not covered by the P7 row/block fusion plan")

        raise ValueError(f"unsupported logical uop[{idx}] opcode={uop.opcode}")

    exec_plan.append(ExecPlanEntry(EXEC_END, 0, len(uops) - 1, 0))
    if len(fixed_exec_descs) & 1:
        # HLS derives fixed_exec_desc_count from fixed_exec/exec_plan offsets.
        # Keep the 64B section alignment explicit instead of letting padding
        # bytes look like an accidental descriptor.
        fixed_exec_descs.append(
            FixedExecDesc(
                kind=EXEC_NOP,
                src0_tensor=TID_INVALID,
                src1_tensor=TID_INVALID,
                dst_tensor=TID_INVALID,
                param_id=0,
            )
        )
    if len(exec_plan) > MAX_EXEC_PLAN_COUNT:
        raise ValueError(f"exec_plan_count={len(exec_plan)} exceeds MAX_EXEC_PLAN_COUNT={MAX_EXEC_PLAN_COUNT}")
    if len(fixed_exec_descs) > MAX_FIXED_EXEC_DESC_COUNT:
        raise ValueError(
            f"fixed_exec_desc_count={len(fixed_exec_descs)} exceeds MAX_FIXED_EXEC_DESC_COUNT={MAX_FIXED_EXEC_DESC_COUNT}"
        )
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
        "fusion_audit": fusion_audit,
        "store_layouts": {
            STORE_LAYOUT_NAMES.get(layout, str(layout)): sum(1 for rc in row_consumers if rc.reserved0 == layout)
            for layout in sorted({rc.reserved0 for rc in row_consumers})
        },
    }
    if coverage["standalone_add"] != 0:
        raise ValueError(f"ADD fusion incomplete: {coverage}")
    return conv_exec_descs, row_consumers, fixed_exec_descs, exec_plan, block5_sched_descs, coverage


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


def tensor_storage_bytes(desc: TensorDesc) -> int:
    phys_c = desc.phys_c if desc.phys_c else desc.c
    return int(desc.h) * int(desc.w) * int(phys_c)


def build_memory_lifetime_audit(uops: Sequence[Uop]) -> dict:
    first_write: Dict[int, int] = {}
    last_write: Dict[int, int] = {}
    first_read: Dict[int, int] = {}
    last_read: Dict[int, int] = {}

    for idx, uop in enumerate(uops):
        for tensor_id in (uop.src0, uop.src1):
            if tensor_id == TID_INVALID:
                continue
            first_read.setdefault(int(tensor_id), idx)
            last_read[int(tensor_id)] = idx
        if uop.dst != TID_INVALID:
            tensor_id = int(uop.dst)
            first_write.setdefault(tensor_id, idx)
            last_write[tensor_id] = idx

    def tensor_lifetime(tensor_id: int) -> dict:
        desc = TENSOR_DESC_BY_ID[tensor_id]
        return {
            "tensor_id": tensor_id,
            "name": desc.name,
            "bank": int(desc.bank_id),
            "base_offset": int(desc.base_offset),
            "bytes": tensor_storage_bytes(desc),
            "first_write": first_write.get(tensor_id),
            "last_write": last_write.get(tensor_id),
            "first_read": first_read.get(tensor_id),
            "last_read": last_read.get(tensor_id),
        }

    pool1 = tensor_lifetime(1)
    pool2 = tensor_lifetime(8)
    pool2_alias_fits = (
        FMBUF_POOL2_ALIAS_BASE == FMBUF_POOL1_BASE
        and FMBUF_POOL2_ALIAS_BYTES <= pool1["bytes"]
        and FMBUF_POOL2_ALIAS_BASE + FMBUF_POOL2_ALIAS_BYTES <= FMBUF_BYTES
    )
    pool2_alias_lifetime_safe = (
        pool1["last_read"] is not None
        and pool2["first_write"] is not None
        and int(pool1["last_read"]) < int(pool2["first_write"])
    )
    pool2_alias_safe = pool2_alias_fits and pool2_alias_lifetime_safe
    if not pool2_alias_safe:
        raise ValueError(
            "unsafe BRAM SCR1 pool2 alias: "
            f"pool1={pool1} pool2={pool2} fits={pool2_alias_fits}"
        )

    return {
        "format": "ESP_INT8_MEMORY_LIFETIME_AUDIT_V1",
        "aliases": [
            {
                "name": "POOL2_OVER_POOL1_FMBUF",
                "virtual_bank": "BANK_BRAM_SCR1",
                "virtual_bank_id": BANK_BRAM_SCR1,
                "target_bank": "BANK_FMEM0",
                "target_bank_id": BANK_FMEM0,
                "base_offset": FMBUF_POOL2_ALIAS_BASE,
                "bytes": FMBUF_POOL2_ALIAS_BYTES,
                "overlays_tensor": "T_POOL1",
                "producer_tensor": "T_POOL2",
                "reason": "T_POOL1 is last read before T_POOL2 is first written; pool temp is not aliased.",
            }
        ],
        "tensor_lifetimes": {
            str(tensor_id): tensor_lifetime(tensor_id)
            for tensor_id in sorted(TENSOR_DESC_BY_ID)
        },
        "hard_checks": {
            "pool2_alias_fits_pool1_region": pool2_alias_fits,
            "pool2_alias_lifetime_safe": pool2_alias_lifetime_safe,
            "pool2_alias_safe": pool2_alias_safe,
        },
    }


def build_param_audit(
    *,
    qparams: Dict[str, Tuple[float, int, str]],
    scale_names: Sequence[str],
    section_offsets: Dict[str, int],
    section_sizes: Dict[str, int],
    window_audit: Sequence[dict],
    weight_report: Sequence[dict],
    fusion_coverage: dict,
    block5_sched_descs: Sequence[Block5SchedDesc],
    memory_audit: dict,
    warnings: Sequence[str],
) -> dict:
    valid_3x3_modes = {
        WIN_MODE_3X3_STAGED_C3,
        WIN_MODE_3X3_STAGED_C12,
        WIN_MODE_3X3_STAGED_C19,
        WIN_MODE_3X3_STAGED_C25,
        WIN_MODE_3X3_STAGED_C28,
        WIN_MODE_3X3_STAGED_C64,
        WIN_MODE_3X3_STAGED_C128,
        WIN_MODE_3X3_STAGED_C131,
    }
    if any(int(x["key"][0]) == 3 and int(x["key"][2]) not in valid_3x3_modes for x in window_audit):
        raise ValueError("PARAM v4 requires every 3x3 schedule to use a supported staged mode")
    if any(int(x["key"][0]) == 3 and int(x.get("cache_chunks", 0)) <= 0 for x in window_audit):
        raise ValueError("PARAM v4 staged 3x3 schedule is missing cache_chunks")
    if any(x["lane_level_suspected"] for x in window_audit):
        raise ValueError("lane-level window schedule suspected; refusing PARAM v4 export")
    if fusion_coverage["add_fused"] != fusion_coverage["add_total"]:
        raise ValueError(f"ADD fusion incomplete: {fusion_coverage}")
    store_layouts = fusion_coverage.get("store_layouts", {})
    if store_layouts.get(STORE_LAYOUT_NAMES[STORE_LAYOUT_NARROW_FIXED], 0):
        raise ValueError(f"S5 hot path selected narrow fixed RMW layout: {store_layouts}")
    if store_layouts.get(STORE_LAYOUT_NAMES[STORE_LAYOUT_COLD_RMW_FALLBACK], 0):
        raise ValueError(f"S5 hot path selected cold RMW fallback: {store_layouts}")
    block5_feasibility = build_block5_feasibility_audit()

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
        "format": "ESP_INT8_PARAM_AUDIT_V4",
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
        "block5_schedule": [
            {
                "pattern": desc.pattern,
                "branch_count": desc.branch_count,
                "first_branch_conv_id": desc.first_branch_conv_id,
                "first_window_sched_id": desc.first_window_sched_id,
                "first_conv_qparam_id": desc.first_conv_qparam_id,
                "src_tensor": desc.src_tensor,
                "dst_tensor": desc.dst_tensor,
                "add_tensor": desc.add_tensor,
                "chain_add_qparam_ids": [
                    desc.chain_add_qparam_id0,
                    desc.chain_add_qparam_id1,
                    desc.chain_add_qparam_id2,
                ],
                "residual_add_qparam_id": desc.residual_add_qparam_id,
                "affine_param_id": desc.affine_param_id,
                "affine_block_count": desc.affine_block_count,
                "finalizer_kind": desc.finalizer_kind,
                "out_h": desc.out_h,
                "out_w": desc.out_w,
                "row_group_h": desc.row_group_h,
                "valid_c": desc.valid_c,
            }
            for desc in block5_sched_descs
        ],
        "block5_feasibility": block5_feasibility,
        "memory_lifetime": memory_audit,
        "hard_checks": {
            "all_3x3_windows_param_scheduled": True,
            "all_3x3_windows_staged": all(
                int(x["key"][2]) in valid_3x3_modes
                for x in window_audit
                if int(x["key"][0]) == 3
            ),
            "hls_does_not_load_window_pack_cmd": True,
            "window_pack_cmd_section_empty": sum(int(x["cmd_count"]) for x in window_audit) == 0,
            "no_lane_level_window_schedule": True,
            "all_adds_fused": True,
            "no_standalone_concat": fusion_coverage["standalone_concat"] == 0,
            "packed_weight_fits_wbuf": sum(int(x["packed_words"]) for x in weight_report) * TK <= WBUF_BYTES,
            "pool2_alias_lifetime_safe": bool(
                memory_audit.get("hard_checks", {}).get("pool2_alias_lifetime_safe", False)
            ),
            "pool2_alias_safe": bool(memory_audit.get("hard_checks", {}).get("pool2_alias_safe", False)),
            "store_layouts_scheduled": bool(store_layouts),
            "no_store_cold_rmw_fallback": store_layouts.get(STORE_LAYOUT_NAMES[STORE_LAYOUT_COLD_RMW_FALLBACK], 0) == 0,
            "block5_schedule_section_present": len(block5_sched_descs) > 0,
        },
        "warnings": list(warnings),
    }


def build_blob_v4(args: argparse.Namespace) -> dict:
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
    memory_audit = build_memory_lifetime_audit(uops)
    window_scheds, window_cmds, schedule_ids_by_param, window_audit = build_window_schedule_sections(uops)
    conv_exec_descs, row_consumers, fixed_exec_descs, exec_plan, block5_sched_descs, fusion_coverage = build_exec_plan_sections(
        uops,
        schedule_ids_by_param,
        packed_weight_word_offsets,
        weight_report,
    )
    rmw_audit = build_rmw_audit(
        uops=uops,
        conv_exec_descs=conv_exec_descs,
        row_consumers=row_consumers,
        exec_plan=exec_plan,
    )
    (out_dir / "rmw_audit.json").write_text(
        json.dumps(rmw_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if rmw_audit["problem_count"] and not args.allow_hot_rmw:
        raise ValueError(
            "P7F export rejected hot-path RMW layouts; see rmw_audit.json. "
            f"layout_counts={rmw_audit['layout_counts']} class_counts={rmw_audit['class_counts']}. "
            "Use --allow-hot-rmw only for legacy/debug artifact reproduction."
        )

    tensor_desc_data = b"".join(pack_tensor_desc(desc) for desc in TENSOR_PLAN)
    scale_desc_data = b"".join(pack_scale_desc(qparams[name][0]) for name in scale_names)
    conv_exec_data = b"".join(pack_conv_exec_desc(desc) for desc in conv_exec_descs)
    window_sched_data = b"".join(pack_window_sched_desc(desc) for desc in window_scheds)
    window_cmd_data = b"".join(pack_window_pack_cmd(cmd) for cmd in window_cmds)
    row_consumer_data = b"".join(pack_row_consumer_desc(desc) for desc in row_consumers)
    fixed_exec_data = b"".join(pack_fixed_exec_desc(desc) for desc in fixed_exec_descs)
    exec_plan_data = b"".join(pack_exec_plan_entry(entry) for entry in exec_plan)
    block5_sched_padded = list(block5_sched_descs) + [
        Block5SchedDesc(
            pattern=BLOCK5_PATTERN_INVALID,
            branch_count=0,
            first_branch_conv_id=0,
            first_window_sched_id=0,
            first_conv_qparam_id=0,
            src_tensor=TID_INVALID,
            dst_tensor=TID_INVALID,
            add_tensor=TID_INVALID,
            chain_add_qparam_id0=0,
            chain_add_qparam_id1=0,
            chain_add_qparam_id2=0,
            residual_add_qparam_id=0,
            affine_param_id=0,
            affine_block_count=0,
            finalizer_kind=BLOCK5_FINALIZER_INVALID,
            scratch_region=0,
            out_h=0,
            out_w=0,
            row_group_h=0,
            valid_c=0,
        )
        for _ in range(MAX_BLOCK5_SCHED_COUNT - len(block5_sched_descs))
    ]
    block5_sched_data = b"".join(pack_block5_sched_desc(desc) for desc in block5_sched_padded)
    uop_data = b"".join(pack_uop(uop) for uop in uops)

    section_data = {
        "tensor_desc": tensor_desc_data,
        "scale_desc": scale_desc_data,
        "conv_exec_desc": conv_exec_data,
        "window_sched_desc": window_sched_data,
        "window_pack_cmd": window_cmd_data,
        "row_consumer_desc": row_consumer_data,
        "fixed_exec_desc": fixed_exec_data,
        "exec_plan": exec_plan_data,
        "block5_sched_desc": block5_sched_data,
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
        offsets["fixed_exec_desc"],
        offsets["exec_plan"],
        len(window_scheds),
        len(window_cmds),
        offsets["block5_sched_desc"],
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
        block5_sched_descs=block5_sched_descs,
        memory_audit=memory_audit,
        warnings=warnings,
    )

    blob_path = out_dir / "param_blob.bin"
    blob_path.write_bytes(blob)
    shutil.copy2(blob_path, out_dir / "PARAM.BIN")
    (out_dir / "uop_table.bin").write_bytes(uop_data)
    (out_dir / "tensor_desc_table.bin").write_bytes(tensor_desc_data)
    (out_dir / "exec_plan.bin").write_bytes(exec_plan_data)
    (out_dir / "fixed_exec_desc.bin").write_bytes(fixed_exec_data)
    (out_dir / "window_pack_cmd.bin").write_bytes(window_cmd_data)
    (out_dir / "block5_sched_desc.bin").write_bytes(block5_sched_data)

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
    (out_dir / "memory_lifetime_audit.json").write_text(
        json.dumps(memory_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "fusion_audit.json").write_text(
        json.dumps(fusion_coverage.get("fusion_audit", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "rmw_audit.json").write_text(
        json.dumps(rmw_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "format": "ESP_INT8_HW_ARTIFACT_SINGLE_V4",
        "artifact_dir": str(artifact_dir),
        "param_blob": str(blob_path),
        "param_bin": str(out_dir / "PARAM.BIN"),
        "param_blob_bytes": len(blob),
        "param_blob_sha256": audit["param_blob"]["sha256"],
        "param_version": PARAM_BLOB_VERSION_SCHED,
        "hardware_contract": {
            "memory_layout": "p7_ppu_bram_pool2_alias_wbuf120k_20260707",
            "schedule": "P7_PARAM_V4_EXEC_PLAN",
            "weight_layout": "SA-ready packed weight words: oc-local TM lane, kt K-tile",
            "window_schedule": "PARAM v4 staged 3x3/1x1 descriptors; pack commands are audit/replay-only",
            "fixed_ops": "PARAM v4 fixed_exec_desc section; no HLS legacy-uop reconstruction",
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
        "fixed_exec_desc_count": len(fixed_exec_descs),
        "block5_sched_desc_count": len(block5_sched_descs),
        "section_offsets": offsets,
        "section_sizes": section_sizes,
        "frame_data": frame_report,
        "conv_qparams": conv_report,
        "affine_qparams": affine_report,
        "add_qparams": add_report,
        "pool_qparams": pool_report,
        "fusion_coverage": fusion_coverage,
        "rmw_audit": {
            "file": str(out_dir / "rmw_audit.json"),
            "problem_count": rmw_audit["problem_count"],
            "layout_counts": rmw_audit["layout_counts"],
            "class_counts": rmw_audit["class_counts"],
        },
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
        parse_and_check_blob_v4(blob_path, audit)
    return manifest


def parse_and_check_blob_v4(path: Path, audit: dict | None = None) -> None:
    data = path.read_bytes()
    if len(data) < PARAM_HEADER_BYTES:
        raise ValueError("PARAM v4 blob is shorter than header")
    header = struct.unpack("<32I", data[:PARAM_HEADER_BYTES])
    if header[0] != PARAM_BLOB_MAGIC or header[1] != PARAM_BLOB_VERSION_SCHED:
        raise ValueError(f"bad PARAM v4 magic/version: magic=0x{header[0]:08x} version={header[1]}")

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
        "fixed_exec_desc": header[26],
        "exec_plan": header[27],
        "block5_sched_desc": header[30],
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
    fixed_off = offsets["fixed_exec_desc"]
    if offsets["exec_plan"] < fixed_off or (offsets["exec_plan"] - fixed_off) % 32 != 0:
        raise ValueError("fixed_exec_desc section must be directly before exec_plan and 32B-aligned")
    fixed_count = (offsets["exec_plan"] - fixed_off) // 32
    if fixed_count == 0 or fixed_count > MAX_FIXED_EXEC_DESC_COUNT:
        raise ValueError(f"invalid fixed_exec_desc_count={fixed_count}")
    if exec_count == 0 or exec_count > MAX_EXEC_PLAN_COUNT:
        raise ValueError(f"invalid exec_entry_count={exec_count}")
    block5_off = offsets["block5_sched_desc"]
    block5_records = data[block5_off : block5_off + MAX_BLOCK5_SCHED_COUNT * 32]
    if len(block5_records) != MAX_BLOCK5_SCHED_COUNT * 32:
        raise ValueError("block5_sched_desc section is truncated")
    active_block5 = 0
    for sid in range(MAX_BLOCK5_SCHED_COUNT):
        bo = block5_off + sid * 32
        pattern, branch_count, first_conv, first_sched, _first_qparam = struct.unpack("<5B", data[bo : bo + 5])
        if pattern == BLOCK5_PATTERN_INVALID:
            continue
        active_block5 += 1
        if pattern not in (BLOCK5_PATTERN_L2_C16_4C12, BLOCK5_PATTERN_L3_C28_4C25):
            raise ValueError(f"block5_sched[{sid}] bad pattern={pattern}")
        if branch_count != 5:
            raise ValueError(f"block5_sched[{sid}] branch_count={branch_count}, expected 5")
        if first_conv + 4 >= counts["conv_exec_desc_count"]:
            raise ValueError(f"block5_sched[{sid}] branch conv range out of bounds")
        if first_sched >= counts["window_sched_count"]:
            raise ValueError(f"block5_sched[{sid}] bad first_window_sched_id={first_sched}")
    if active_block5 == 0:
        raise ValueError("PARAM v4 is missing active block5_sched_desc records")
    last = struct.unpack("<4B", data[exec_off + (exec_count - 1) * 4 : exec_off + exec_count * 4])
    if last[0] != EXEC_END:
        raise ValueError(f"exec_plan last entry is not EXEC_END: {last}")
    executed_conv_desc_ids: set[int] = set()
    for i in range(exec_count):
        kind, desc_id, _logical, _flags = struct.unpack("<4B", data[exec_off + i * 4 : exec_off + (i + 1) * 4])
        if kind not in (EXEC_CONV, EXEC_POOL, EXEC_BLOCK_AFFINE, EXEC_BLOCK_ADD_AFFINE, EXEC_END):
            raise ValueError(f"unsupported exec_plan[{i}] kind={kind}")
        if kind == EXEC_CONV:
            executed_conv_desc_ids.add(desc_id)
        if kind in (EXEC_POOL, EXEC_BLOCK_AFFINE, EXEC_BLOCK_ADD_AFFINE):
            if desc_id >= fixed_count:
                raise ValueError(f"exec_plan[{i}] fixed desc_id={desc_id} out of range {fixed_count}")
            fixed_kind = data[fixed_off + desc_id * 32]
            if fixed_kind != kind:
                raise ValueError(f"exec_plan[{i}] kind={kind} but fixed_desc[{desc_id}].kind={fixed_kind}")

    cmd_count = counts["window_pack_cmd_count"]
    sched_count = counts["window_sched_count"]
    sched_off = offsets["window_sched_desc"]
    schedule_modes: List[int] = []
    for sid in range(sched_count):
        off = sched_off + sid * 128
        fields = struct.unpack("<8B5H", data[off : off + 18])
        mode, kernel, stride, dilation, padding, cache_chunks, _cache_col_slots, _flags, in_c, out_w, k_tiles, cmd_base, cmd_len = fields
        schedule_modes.append(mode)
        if mode == WIN_MODE_INVALID or kernel not in (1, 3) or in_c == 0:
            raise ValueError(f"bad window schedule[{sid}] header: {fields}")
        if stride == 0 or out_w == 0:
            raise ValueError(f"window schedule[{sid}] has invalid stride/out_w: {fields}")
        if kernel == 3:
            valid_3x3_modes = {
                WIN_MODE_3X3_STAGED_C3,
                WIN_MODE_3X3_STAGED_C12,
                WIN_MODE_3X3_STAGED_C19,
                WIN_MODE_3X3_STAGED_C25,
                WIN_MODE_3X3_STAGED_C28,
                WIN_MODE_3X3_STAGED_C64,
                WIN_MODE_3X3_STAGED_C128,
                WIN_MODE_3X3_STAGED_C131,
            }
            if mode not in valid_3x3_modes:
                raise ValueError(f"window schedule[{sid}] kernel=3 uses unsupported mode={mode}")
            if cache_chunks <= 0 or cache_chunks > 5:
                raise ValueError(f"window schedule[{sid}] invalid cache_chunks={cache_chunks}")
        if k_tiles > MAX_K_TILE_COUNT:
            raise ValueError(f"window schedule[{sid}] k_tiles={k_tiles} exceeds max")
        if cmd_base + cmd_len > cmd_count:
            raise ValueError(f"window schedule[{sid}] command range out of bounds")

    row_consumer_off = offsets["row_consumer_desc"]
    row_consumer_count = counts["row_consumer_desc_count"]
    row_consumer_modes: List[int] = []
    row_consumer_layouts: List[int] = []
    ppu_allowed_layouts = {
        STORE_LAYOUT_NONE,
        STORE_LAYOUT_COMPACT_C12,
        STORE_LAYOUT_COMPACT_C19,
        STORE_LAYOUT_COMPACT_C25,
    }
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
        if mode in (ROW_CONSUMER_NONE, ROW_CONSUMER_CAT_AFFINE_STORE):
            if mode == ROW_CONSUMER_NONE and layout == STORE_LAYOUT_NONE:
                continue
            if layout == STORE_LAYOUT_NONE:
                raise ValueError(f"row_consumer[{rid}] mode={mode} is missing scheduled store layout")
            if layout not in ppu_allowed_layouts:
                name = STORE_LAYOUT_NAMES.get(layout, str(layout))
                raise ValueError(
                    f"row_consumer[{rid}] selected unsupported PPU-1 store layout={name}"
                )
        else:
            name = ROW_CONSUMER_NAMES.get(mode, str(mode))
            raise ValueError(f"row_consumer[{rid}] unsupported PPU-1 mode={name}")

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
        if (
            cid in executed_conv_desc_ids
            and row_consumer_modes[row_consumer_id] != ROW_CONSUMER_UPSAMPLE_OUT
            and row_consumer_layouts[row_consumer_id] == STORE_LAYOUT_NONE
        ):
            raise ValueError(f"executed conv_exec[{cid}] row_consumer[{row_consumer_id}] has no store layout")
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
        if not audit.get("hard_checks", {}).get("pool2_alias_safe", False):
            raise ValueError("audit reports unsafe pool2 BRAM alias")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        default=r"D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_3ep",
        help="Existing quantized artifact directory produced from D:\\ESPNet.",
    )
    parser.add_argument(
        "--out-dir",
        default=r"D:\ESP_INT8\hw_artifacts\sched_v4_p7_0702",
        help="Hardware-facing output directory.",
    )
    parser.add_argument(
        "--param-version",
        type=int,
        choices=(4,),
        default=4,
        help="Only PARAM v4/P7 staged-WinGen schedule blobs are supported.",
    )
    parser.add_argument(
        "--strict-zp",
        action="store_true",
        help="Fail if non-zero activation zp or bypass scale mismatches are detected.",
    )
    parser.add_argument(
        "--strict-schedule",
        action="store_true",
        help="Fail if PARAM v4 schedule, weight packing, or fusion audit is incomplete.",
    )
    parser.add_argument(
        "--allow-hot-rmw",
        action="store_true",
        help="Legacy/debug only: allow PARAM v4 export to contain hot-path PREFIX/NARROW/COLD_RMW layouts.",
    )
    args = parser.parse_args()

    manifest = build_blob_v4(args)
    parse_and_check_blob_v4(Path(manifest["param_blob"]), load_json(Path(args.out_dir) / "param_audit.json"))

    print(f"Exported param blob: {manifest['param_blob']}")
    print(f"Output directory: {args.out_dir}")
    print(f"PARAM version: {manifest.get('param_version', PARAM_BLOB_VERSION_SCHED)}")
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
