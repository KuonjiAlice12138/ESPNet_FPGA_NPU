from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import export_int8_hw_blob as exporter
from tools import analyze_sa_utilization as model
from tools.hw_param_replay import (
    ConvExecDesc, ParamBlob, TensorDesc, WindowSchedDesc,
    window_row_reuse_contract_valid,
)


def reuse_word(words: int, mode: int = 2) -> int:
    return mode | (words << 2)


def reuse_contract(width: int = 512):
    sched = WindowSchedDesc(mode=4, kernel=3, stride=2, dilation=1,
                            padding=1, in_c=3, out_w=width // 2,
                            flags=1, loader_class=1, k_tiles=1, reserved=[0] * 10)
    sched.reserved[9] = reuse_word(width * 3 // 32)
    conv = ConvExecDesc(in_h=256, in_w=width, in_c=3, out_c=16,
                        kernel=3, stride=2, dilation=1, padding=1, k_tiles=1)
    src = TensorDesc(h=256, w=width, c=3)
    return sched, conv, src


class C3RowReuseTests(unittest.TestCase):
    def test_compiler_uses_length_not_capacity(self):
        for width, words in ((512, 48), (1024, 96)):
            self.assertEqual(exporter.compile_window_row_reuse_word(
                source_w=width, in_c=3, kernel=3, stride=2, dilation=1,
                loader_class=1, flags=1), reuse_word(words))

    def test_c12_is_not_reenabled(self):
        for width in (128, 256):
            self.assertEqual(exporter.compile_window_row_reuse_word(
                source_w=width, in_c=12, kernel=3, stride=1, dilation=1,
                loader_class=1, flags=1), 0)
        for width in (0, 513, 1056):
            with self.assertRaises(ValueError):
                exporter.compile_window_row_reuse_word(
                    source_w=width, in_c=3, kernel=3, stride=2, dilation=1,
                    loader_class=1, flags=1)

    def test_physical_layout_and_word_count_are_checked(self):
        for width in (512, 1024):
            sched, conv, src = reuse_contract(width)
            self.assertTrue(window_row_reuse_contract_valid(sched, conv, src))
            self.assertFalse(window_row_reuse_contract_valid(sched))
            for words in (0, 49, 97):
                bad = dataclasses.replace(sched, reserved=[0] * 9 + [reuse_word(words)])
                self.assertFalse(window_row_reuse_contract_valid(bad, conv, src))
            for bad_src in (dataclasses.replace(src, reserved0=4),
                            dataclasses.replace(src, reserved1=1),
                            dataclasses.replace(src, base_offset=1),
                            dataclasses.replace(src, w=width - 1)):
                self.assertFalse(window_row_reuse_contract_valid(sched, conv, bad_src))
            for field, value in (("mode", 5), ("stride", 1), ("dilation", 2),
                                 ("flags", 0), ("in_c", 12), ("out_w", width // 2 - 1)):
                self.assertFalse(window_row_reuse_contract_valid(
                    dataclasses.replace(sched, **{field: value}), conv, src))
            for word in (reuse_word(48, 1), reuse_word(48, 3), reuse_word(48) | 512, 48 << 2):
                self.assertFalse(window_row_reuse_contract_valid(
                    dataclasses.replace(sched, reserved=[0] * 9 + [word]), conv, src))

    def test_blob_validates_schedule_against_each_source(self):
        path = ROOT / "hw_artifacts/binary2_int8_h256w512_r2_v4/PARAM.BIN"
        blob = ParamBlob(path)
        conv = blob.conv_exec_by_index[0]
        schedule_offset = blob.header["window_sched_offset"] + 128 * conv.window_sched_id
        source_offset = blob.header["tensor_desc_offset"] + 16 * conv.src_tensor
        data = bytearray(path.read_bytes())
        struct.pack_into("<H", data, schedule_offset + 126, reuse_word(48))
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "PARAM.BIN"
            candidate.write_bytes(data)
            self.assertEqual(ParamBlob(candidate).window_sched[conv.window_sched_id].row_reuse_mode, 2)
            struct.pack_into("<H", data, source_offset + 2, 4)
            candidate.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "row-reuse"):
                ParamBlob(candidate)

    def test_schedule_recompile_preserves_every_numeric_byte(self):
        for name in ("binary2", "cityscapes20"):
            path=ROOT/f"hw_artifacts/{name}_int8_h256w512_r2_v4/PARAM.BIN"
            original=path.read_bytes(); blob=ParamBlob(path)
            compiled=exporter.compile_window_reuse_param(path)
            allowed=set()
            for sid in range(len(blob.window_sched_by_index)):
                offset=blob.header["window_sched_offset"]+128*sid+126
                allowed.update((offset,offset+1))
            self.assertTrue(all(a==b or i in allowed for i,(a,b) in enumerate(zip(original,compiled))))
            self.assertEqual(compiled[blob.header["weight_packed_offset"]:blob.header["uop_offset"]],
                             original[blob.header["weight_packed_offset"]:blob.header["uop_offset"]])

    def test_unified_rgb_publication_keeps_frozen_inputs_and_logits(self):
        for key,name in (("R256B2","binary2"),("R256B20","cityscapes20")):
            frozen=ROOT/f"hw_artifacts/{name}_int8_h256w512_r2_v4"
            candidate=ROOT/f"hw_artifacts/unified_int8/{key}"
            manifest=exporter.validate_runtime_artifact(candidate)
            self.assertEqual((candidate/"INPUTQ.BIN").read_bytes(),(frozen/"INPUTQ.BIN").read_bytes())
            self.assertEqual((candidate/"LOGITS.BIN").read_bytes(),(frozen/"golden_output_q.bin").read_bytes())
            self.assertEqual(manifest["provenance"]["replay_mismatches"],0)

    def test_service_model_counts_drain_per_issue_and_per_row(self):
        uop = model.Uop(2, 2, 0, 0, 0, 0, 0, 0, 256, 512, 3, 16,
                        3, 2, 1, 1, 0, 16, 0)
        row = model.conv_stats(uop, 32, 32, pixel_parallel=True)
        service = model.pipeline_service_model(row, sa_k_iteration_latency=35,
                                               post_iteration_latency=4)
        self.assertEqual(service["output_issues"], 16384)
        self.assertEqual(service["sa_drain_cycles"], 34 * 16384)
        self.assertEqual(service["post_drain_cycles"], 3 * 128)
        self.assertEqual(service["sa_service_cycles_no_stall"],
                         16384 * (1 + 2 + 34 + 1))
        self.assertLess(service["ideal_row_dataflow_cycles_lower_bound"],
                        service["pipeline_service_floor_cycles_no_stall"])

    def test_c3_source_reads_are_not_activation_tokens(self):
        artifact = ROOT / "hw_artifacts/binary2_int8_h256w512_r2_v4"
        blob = ParamBlob(artifact / "PARAM.BIN")
        conv = blob.conv_exec_by_index[0]
        sched = blob.window_sched[conv.window_sched_id]
        reads = model.c3_row_read_model(conv, sched, blob.tensor_desc[conv.src_tensor])
        self.assertEqual(reads["source_aligned_reads_none"], 18384)
        self.assertEqual(reads["source_aligned_reads_keep1"], 12288)

    def test_source_read_model_respects_padding_geometry(self):
        sched, conv, src = reuse_contract()
        sched = dataclasses.replace(sched, padding=0, out_w=255, flags=3)
        conv = dataclasses.replace(conv, in_h=8, padding=0)
        src = dataclasses.replace(src, h=8)
        reads = model.c3_row_read_model(conv, sched, src)
        # Three valid output rows consume source rows 0..6, not row 7.
        self.assertEqual(reads["source_aligned_reads_keep1"], 7 * 48)


if __name__ == "__main__":
    unittest.main()
