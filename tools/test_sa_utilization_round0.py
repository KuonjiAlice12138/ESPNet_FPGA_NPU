from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

from tools import analyze_sa_utilization as model
from tools.hw_param_replay import ParamBlob


ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "tools/testdata/rtl_stage_0910.csv"


class ScheduledUtilizationTests(unittest.TestCase):
    def test_paired_odd_width_counts_issues_not_pixels(self) -> None:
        uop = model.Uop(0, 2, 0, 0, 0, 1, 0, 0, 1, 5, 12, 12,
                        1, 1, 1, 0, 0, 12, 0)
        row = model.conv_stats(uop, 32, 32, pixel_parallel=True)
        self.assertEqual(row.issue_count_per_row, 3)
        self.assertEqual(row.pe_slot_macs, 3 * 32 * 32)
        self.assertEqual(row.useful_macs, 5 * 12 * 12)

    def test_release_fill_and_token_gates(self) -> None:
        for name, macs, slots, issues, fill in (
            ("binary2", 336435200, 480247808, 468992, 0.7005),
            ("cityscapes20", 345872384, 488636416, 477184, 0.7078),
        ):
            with self.subTest(profile=name):
                artifact = ROOT / f"hw_artifacts/{name}_int8_h256w512_r2_v4"
                rows = model.artifact_conv_stats(artifact, 32, 32)
                report = model.performance_summary(rows, 32, 32)
                self.assertEqual(len(rows), 26)
                self.assertEqual(report["useful_macs"], macs)
                self.assertEqual(report["pe_slot_macs"], slots)
                self.assertEqual(report["activation_k_tile_issues"], issues)
                self.assertAlmostEqual(report["weighted_arithmetic_fill"], fill, places=4)
                self.assertTrue(all(row.schedule_tokens_match for row in rows))
                self.assertEqual(sum(row.act_stream0_words for row in rows), issues)
                self.assertEqual(sum(row.psum_stream_words for row in rows),
                                 sum(row.out_h * row.issue_count_per_row * row.oc_tiles *
                                     row.psum_words_per_issue for row in rows))

    def test_descriptor_mismatch_is_rejected(self) -> None:
        artifact = ROOT / "hw_artifacts/binary2_int8_h256w512_r2_v4"
        blob = ParamBlob(artifact / "PARAM.BIN")
        uop = next(u for u in model.parse_uops(artifact / "uop_table.bin")
                   if u.opcode == model.UOP_CONV)
        desc = blob.conv_exec_by_index[0]
        sched = blob.window_sched[desc.window_sched_id]
        row = model.conv_stats(uop, 32, 32, pixel_parallel=True)
        with self.assertRaisesRegex(ValueError, "out_w"):
            model.validate_schedule_tokens(uop, row, desc,
                                           dataclasses.replace(sched, out_w=sched.out_w + 1))

    def test_measured_metrics_and_quarter_resolution_scaling(self) -> None:
        profiles = model.parse_stage_csv(STAGES)
        for name, total, conv in (("binary2", 13893492, 9139308),
                                  ("cityscapes20", 13911890, 9157708)):
            artifact = ROOT / f"hw_artifacts/{name}_int8_h256w512_r2_v4"
            rows = model.artifact_conv_stats(artifact, 32, 32)
            measured = profiles[name]
            report = model.performance_summary(rows, 32, 32, conv, total)
            self.assertEqual(measured["TOTAL"], total)
            self.assertEqual(sum(v for k, v in measured.items() if k != "TOTAL"), total)
            self.assertAlmostEqual(report["temporal_issue_occupancy"],
                                   report["activation_k_tile_issues"] / conv)
            self.assertAlmostEqual(report["conv_peak_utilization"],
                                   report["useful_macs"] / (conv * 1024))
            self.assertAlmostEqual(report["end_to_end_peak_utilization"],
                                   report["useful_macs"] / (total * 1024))
        scaling = model.scaling_stats(profiles["binary2"], profiles["binary2_h512w1024"], 0.25)
        total_row = next(row for row in scaling if row["stage_name"] == "TOTAL")
        self.assertEqual(total_row["expected_scaled_cycles"], 51319833 / 4)
        self.assertAlmostEqual(total_row["excess_fraction"], 0.0829, places=4)

    def test_exec_membership_is_taken_from_param_not_uop_cutoffs(self) -> None:
        artifact = ROOT / "hw_artifacts/binary2_int8_h256w512_r2_v4"
        rows = model.artifact_conv_stats(artifact, 32, 32)
        execs = model.exec_stats(artifact, rows)
        self.assertEqual(len(execs), 16)
        self.assertEqual([r["exec_index"] for r in execs], list(range(16)))
        self.assertEqual(sum(r["conv_count"] for r in execs), 26)
        self.assertEqual(execs[5]["conv_uops"], [8, 9, 11, 14, 17])
        self.assertEqual(execs[-1]["kind"], "END")
        self.assertTrue(all(r["measured_cycles"] is None for r in execs))
        self.assertEqual(sum(r["useful_macs"] for r in execs), 336435200)

    def test_prefix_metadata_must_match_the_selected_param(self) -> None:
        artifact = ROOT / "hw_artifacts/binary2_int8_h256w512_r2_v4"
        rows = model.artifact_conv_stats(artifact, 32, 32)
        record = {"record": "PREFIX_DELTA", "new_exec_index": "1", "logical_uop": "2",
                  "kind": "1", "rtl_total": "120", "s5_conv": "100"}
        execs = model.exec_stats(artifact, rows, [record])
        self.assertEqual(execs[1]["measured_cycles"], 120)
        self.assertEqual(execs[1]["measured_conv_cycles"], 100)
        record["logical_uop"] = "7"
        with self.assertRaisesRegex(ValueError, "prefix metadata"):
            model.exec_stats(artifact, rows, [record])


if __name__ == "__main__":
    unittest.main()
