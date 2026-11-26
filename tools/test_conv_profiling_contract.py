from __future__ import annotations

import os
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RTL_ROOT = Path(
    os.environ.get(
        "NPU_STAGE_COUNTER_ROOT",
        r"D:\npu_stage_counter\npu_stage_counter.srcs",
    )
)


def read_repo(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


class ConvProfilingContractTests(unittest.TestCase):
    def test_hls_declares_three_bounded_state_domains(self) -> None:
        ctrl = read_repo("ESP_INT8_hls/include/npu_ctrl.hpp")
        required = (
            "PROF_CONV_WIN_IDLE_OR_DONE",
            "PROF_CONV_WIN_ACTIVE",
            "PROF_CONV_WIN_STATE_COUNT",
            "PROF_CONV_SA_IDLE_OR_DONE",
            "PROF_CONV_SA_COMPUTE_OR_WAIT_ACT",
            "PROF_CONV_SA_PSUM_EMIT_OR_WAIT",
            "PROF_CONV_SA_STATE_COUNT",
            "PROF_CONV_POST_IDLE_OR_DONE",
            "PROF_CONV_POST_REQUANT_OR_WAIT_PSUM",
            "PROF_CONV_POST_ROWBUF_WRITE",
            "PROF_CONV_POST_STATE_COUNT",
        )
        for token in required:
            self.assertIn(token, ctrl)

        expected_counts = {
            "PROF_CONV_WIN_STATE_COUNT": 2,
            "PROF_CONV_SA_STATE_COUNT": 3,
            "PROF_CONV_POST_STATE_COUNT": 3,
        }
        for count_name, expected in expected_counts.items():
            match = re.search(rf"{count_name}\s*=\s*(\d+)", ctrl)
            self.assertIsNotNone(match, count_name)
            self.assertEqual(int(match.group(1)), expected, count_name)

    def test_hls_top_exposes_observational_state_ports(self) -> None:
        top = read_repo("ESP_INT8_hls/src/int8_core.cpp")
        for port in (
            "prof_conv_win_state",
            "prof_conv_sa_state",
            "prof_conv_post_state",
        ):
            self.assertIn(f"volatile esp_int8::u8_t& {port}", top)
            self.assertIn(f"#pragma HLS INTERFACE ap_none port={port}", top)

        conv = read_repo("ESP_INT8_hls/src/conv_engine.cpp")
        self.assertEqual(conv.count("systolic_array_core_row("), 2)
        self.assertEqual(conv.count("scheduled_window_generator_row("), 2)
        self.assertNotIn("read_nb(", conv)
        self.assertNotIn("write_nb(", conv)

        top_call = read_repo("ESP_INT8_hls/tb/top_call.hpp")
        for port in (
            "prof_conv_win_state",
            "prof_conv_sa_state",
            "prof_conv_post_state",
        ):
            self.assertGreaterEqual(top_call.count(port), 3)

    def test_rtl_counter_exposes_conv_state_register_banks(self) -> None:
        rtl_path = RTL_ROOT / "sources_1/new/npu_stage_counter.v"
        self.assertTrue(rtl_path.is_file(), rtl_path)
        rtl = rtl_path.read_text(encoding="utf-8")
        for port in (
            "prof_conv_win_state",
            "prof_conv_sa_state",
            "prof_conv_post_state",
        ):
            self.assertRegex(rtl, rf"input\s+wire\s+\[7:0\]\s+{port}")
        for token in (
            "ADDR_CAPABILITY",
            "ADDR_CONV_CURRENT",
            "ADDR_CONV_WIN_BASE",
            "ADDR_CONV_SA_BASE",
            "ADDR_CONV_POST_BASE",
            "conv_win_active_cycles",
            "conv_sa_active_cycles",
            "conv_post_active_cycles",
            "CONV_COUNTER_WIDTH = 54",
        ):
            self.assertIn(token, rtl)

        tb_path = RTL_ROOT / "sim_1/new/tb_npu_stage_counter.v"
        self.assertTrue(tb_path.is_file(), tb_path)
        tb = tb_path.read_text(encoding="utf-8")
        self.assertIn("derived conv win idle is not stored", tb)
        self.assertIn("conv freeze", tb)

    def test_app_profiles_runtime_selected_models_and_rejects_bad_timings(self):
        main = read_repo("ESP_INT8_app/src/main.c")
        driver = read_repo("ESP_INT8_app/src/hal/int8_npu.c")
        self.assertIn("stage_counter_has_conv_profile()", main)
        self.assertIn("profile_valid()", main)
        self.assertIn("runtime_crc32(g_output,item->output_bytes)", main)
        self.assertIn("RTL_CONV_CHECK", driver)
        self.assertIn("STAGING_WORD_OR_WAIT_PSUM", driver)
        self.assertIn("COMMIT_WORD_OR_WAIT_PSUM", driver)
        begin=main.index("start=timer_counter();")
        end=main.index("ticks=timer_counter()-start;",begin)
        self.assertLess(main.index("Xil_DCacheFlushRange"),begin)
        self.assertLess(begin,main.index("int8_npu_run_infer("))
        self.assertLess(main.index("int8_npu_run_infer("),end)
        self.assertLess(end,main.index("stage_counter_dump_csv();"))
        self.assertLess(end,main.index("Xil_DCacheInvalidateRange"))
        self.assertNotIn("ENABLE_DUAL_SINGLE",main)

if __name__ == "__main__":
    unittest.main()
