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
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
FMBUF_BYTES = 0x598000
FMBUF_URAM_BYTES = 0x380000
MAX_BLOCK5_SCHED_COUNT = 8

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
    def __post_init__(self):
        if self.kt_cmd_base is None:
            self.kt_cmd_base = []


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
                reserved0=self._u16(bo + 2),
                base_offset=self._u32(bo + 4),
                h=self._u16(bo + 8),
                w=self._u16(bo + 10),
                c=self._u16(bo + 12),
                reserved1=self._u16(bo + 14),
            ))
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

    def load_tile(self, desc: TensorDesc, h: int, w: int, c_begin: int, count: int) -> np.ndarray:
        if h < 0 or w < 0 or h >= desc.h or w >= desc.w or c_begin >= desc.c:
            return np.zeros(count, dtype=np.int8)
        valid = min(count, desc.c - c_begin, desc.phys_c)
        off = desc.base_offset + desc.elem_offset(h, w, c_begin)
        result = np.zeros(count, dtype=np.int8)
        slice_buf = self.buf[off:off + valid].astype(np.int8)
        result[:valid] = slice_buf
        return result

    def store_tile(self, desc: TensorDesc, h: int, w: int, c_begin: int, count: int, data: np.ndarray):
        if h >= desc.h or w >= desc.w or c_begin >= desc.c:
            return
        valid = min(count, desc.c - c_begin)
        off = desc.base_offset + desc.elem_offset(h, w, c_begin)
        self.buf[off:off + valid] = np.clip(data[:valid], INT8_MIN, INT8_MAX).astype(np.uint8)

    def dump_tensor(self, desc: TensorDesc) -> np.ndarray:
        result = np.zeros((desc.h, desc.w, desc.c), dtype=np.int8)
        for h in range(desc.h):
            for w in range(desc.w):
                off = desc.base_offset + desc.elem_offset(h, w, 0)
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
        start = int(desc.base_offset)
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
    parser.add_argument("--dump-dir", type=Path, required=True,
                        help="Path to CSim dump directory for comparison")
    parser.add_argument("--layer", type=str, default="all",
                        help="Layer to replay: all, U38, U39, U40")
    args = parser.parse_args()

    blob = ParamBlob(args.param)
    print(f"PARAM: exec_plan entries={len(blob.exec_plan)} conv_exec={len(blob.conv_exec)}")

    # Load input
    input_data = np.frombuffer(args.input.read_bytes(), dtype=np.int8)
    input_3c = input_data.reshape(512, 1024, 3)

    # Initialize feature memory
    fmem = FeatureMemory()
    
    # Root tensor descriptors (simplified — full replay would read from blob)
    T_INPUT = 0
    desc_input = TensorDesc(bank_id=0, base_offset=0, h=512, w=1024, c=3)
    # Load input frame
    for h in range(512):
        for w in range(1024):
            off = h * 1024 + w
            fmem.buf[off * 3:(off + 1) * 3] = np.clip(input_3c[h, w, :], INT8_MIN, INT8_MAX).astype(np.uint8)

    # Focus: replay U39 (b2_bn affine) and U40 (level3_0_c1 conv)
    # For these, we need B2_CAT as input. Let's load it from CSim dump.
    dump_dir = args.dump_dir

    if args.layer in ("U39", "all"):
        # Replay U39 affine (b2_bn)
        # Input: B2_CAT at TID=9. Read from CSim dump if available.
        b2cat_path = dump_dir / "csim_u38_b2_cat.bin"
        if b2cat_path.exists():
            b2cat = np.frombuffer(b2cat_path.read_bytes(), dtype=np.int8).reshape(128, 256, 131)
            # Load into fmem as if written by U38 store
            desc_b2cat = TensorDesc(bank_id=0, base_offset=0, h=128, w=256, c=131, reserved0=131)
            for h in range(128):
                for w in range(256):
                    off = desc_b2cat.elem_offset(h, w, 0)
                    fmem.buf[off:off + 131] = np.clip(b2cat[h, w, :], INT8_MIN, INT8_MAX).astype(np.uint8)
            print(f"Loaded B2_CAT from {b2cat_path} → fmem")
        else:
            print(f"WARNING: {b2cat_path} not found, generating random B2_CAT")
            desc_b2cat = TensorDesc(bank_id=0, base_offset=0, h=128, w=256, c=131, reserved0=131)
            for h in range(128):
                for w in range(256):
                    off = desc_b2cat.elem_offset(h, w, 0)
                    fmem.buf[off:off + 131] = np.random.randint(-10, 10, 131, dtype=np.int8).astype(np.uint8)

        # U39: AFFINE (param_id=3, B2_CAT -> B2_ACT)
        # Use HLS affine math directly
        affine_qparam_path = dump_dir.parent / "csim_u71_b3_bn.bin"  # placeholder
        # For U39, we need affine qparam for param_id=3
        # Let's use the same pattern as conv_qparam
        print("Replaying U39 affine...")
        desc_b2act = TensorDesc(bank_id=0, base_offset=0, h=128, w=256, c=131, reserved0=131)
        # Read U39 qparam from blob
        # affine qparam is at different offsets, skip for now
        print("U39: done (affine replay stub)")

    if args.layer in ("U40", "all"):
        desc_u40 = blob.conv_exec.get(13)
        if desc_u40:
            print(f"\nU40 conv_exec_desc: in={desc_u40.in_h}x{desc_u40.in_w}x{desc_u40.in_c} out_c={desc_u40.out_c} k_tiles={desc_u40.k_tiles}")
            src_desc = TensorDesc(bank_id=0, base_offset=0, h=128, w=256, c=131, reserved0=131)
            dst_desc = TensorDesc(bank_id=0, base_offset=0, h=64, w=128, c=25, reserved0=25)
            consumer = blob.row_consumer.get(desc_u40.row_consumer_id, RowConsumerDesc())

            print("Replaying U40 large-C conv (this may take a while)...")
            replay_conv(fmem, blob, 13, src_desc, dst_desc, consumer)

            # Compare with CSim dump
            dump_path = dump_dir / "csim_u40_level3_0_c1.bin"
            if dump_path.exists():
                hls_dump = np.frombuffer(dump_path.read_bytes(), dtype=np.int8).reshape(64, 128, 25)
                replay_dump = fmem.dump_tensor(dst_desc)
                diff = (hls_dump.astype(np.int32) - replay_dump.astype(np.int32)) != 0
                mismatches = diff.sum()
                mae = float(np.abs(hls_dump.astype(np.int32) - replay_dump.astype(np.int32)).mean())
                print(f"U40 replay vs HLS: mismatch={mismatches}/{diff.size} ({100*mismatches/diff.size:.1f}%) mae={mae:.2f}")
                if mismatches > 0:
                    # Show first few diffs
                    idxs = np.where(diff)
                    print("First 5 mismatches:")
                    for i in range(min(5, len(idxs[0]))):
                        h, w, c = idxs[0][i], idxs[1][i], idxs[2][i]
                        print(f"  ({h},{w},c{c}): HLS={hls_dump[h,w,c]} replay={replay_dump[h,w,c]}")
            else:
                print(f"No CSim dump at {dump_path}")


if __name__ == "__main__":
    main()
