from __future__ import annotations

import unittest
from pathlib import Path

from tools.analyze_conv_cycle_profile import (
    build_useful_cycle_model,
    build_attribution_rows,
    parse_prefix_lines,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConvCycleProfileAnalysisTests(unittest.TestCase):
    def test_release_artifact_builds_exec_useful_cycle_model(self) -> None:
        artifact_dir = REPO_ROOT / "hw_artifacts/binary2_int8_h256w512_r2_v4"
        useful = build_useful_cycle_model(artifact_dir)
        self.assertEqual(set(useful), {1, 4, 5, 6, 7, 9, 10, 11, 12, 14})
        for owner_cycles in useful.values():
            self.assertGreater(owner_cycles["win"], 0)
            self.assertGreater(owner_cycles["sa"], owner_cycles["win"])
            self.assertGreater(owner_cycles["post"], 0)

    def test_prefix_delta_is_attributed_without_adding_parallel_owners(self) -> None:
        header = (
            "record,prefix,active_execs,new_exec_index,logical_uop,kind,"
            "arm_ticks,rtl_total,rtl_window,"
            "s0_idle,s1_param,s2_frame,s3_ctrl,s4_wgt,s5_conv,s6_ppu,"
            "s7_b5,s8_vec,s9_pool,s10_up,s11_store,s12_error,conv_cap,"
            "cw0_idle_done,cw1_active,"
            "cs0_idle_done,cs1_compute_wait_act,cs2_emit_wait,"
            "cp0_idle_done,cp1_requant_wait,cp2_write,"
            "conv_current,current,status"
        )
        delta = (
            "PREFIX_DELTA,P02,2,1,2,1,100,12,12,"
            "0,0,0,0,0,12,0,0,0,0,0,0,0,1,"
            "0,12,2,7,3,4,6,2,"
            "0x00020201,0x00000000,0x00000005"
        )

        records = parse_prefix_lines([header, delta])
        rows = build_attribution_rows(
            records,
            useful_by_exec={1: {"win": 10, "sa": 7, "post": 7}},
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["conv_cycles"], 12)
        self.assertEqual(row["win_sum"], 12)
        self.assertEqual(row["sa_sum"], 12)
        self.assertEqual(row["post_sum"], 12)
        self.assertTrue(row["owner_sums_match"])
        self.assertEqual(row["win_active"], 12)
        self.assertEqual(row["sa_active"], 10)
        self.assertEqual(row["post_active"], 8)
        self.assertAlmostEqual(row["sa_amplification"], 10.0 / 7.0)
        self.assertEqual(row["limiter_owner"], "UNRESOLVED")
        self.assertEqual(row["largest_excess_owner"], "SA")
        self.assertEqual(row["attribution_status"], "state_residency_not_causal_stall")

    def test_non_conv_prefix_keeps_zero_owner_sums(self) -> None:
        header = (
            "record,prefix,active_execs,new_exec_index,logical_uop,kind,"
            "s5_conv,conv_cap,cw0_idle_done,cw1_active,"
            "cs0_idle_done,cs1_compute_wait_act,cs2_emit_wait,"
            "cp0_idle_done,cp1_requant_wait,cp2_write"
        )
        delta = "PREFIX_DELTA,P01,1,0,1,2,0,1,0,0,0,0,0,0,0,0"
        rows = build_attribution_rows(parse_prefix_lines([header, delta]))
        self.assertEqual(rows[0]["conv_cycles"], 0)
        self.assertTrue(rows[0]["owner_sums_match"])
        self.assertEqual(rows[0]["limiter_owner"], "NONE")


if __name__ == "__main__":
    unittest.main()
