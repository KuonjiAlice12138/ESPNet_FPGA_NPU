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
    def test_release_requires_v4_and_conv_profile_counter(self) -> None:
        cfg = read_repo("ESP_INT8_app/src/app_config.h")
        main = read_repo("ESP_INT8_app/src/main.c")
        driver = read_repo("ESP_INT8_app/src/hal/int8_npu.c")
        self.assertRegex(cfg, r"#define INT8_PARAM_BLOB_VERSION_CURRENT 4U")
        self.assertRegex(cfg, r"#define INT8_EXPECTED_EXEC_PLAN_COUNT 16U")
        self.assertRegex(cfg, r"#define INT8_APP_REQUIRE_CONV_PROFILE 1U")
        self.assertIn("if (INT8_APP_REQUIRE_CONV_PROFILE && conv_profile_cap == 0U)", main)
        self.assertNotIn("INT8_PARAM_BLOB_VERSION_LEGACY", driver)
        self.assertNotIn("INT8_EXPECTED_EXEC_PLAN_COUNT_V5", main + driver)

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

    def test_app_reads_capability_and_prints_conv_cycle_checks(self) -> None:
        header = read_repo("ESP_INT8_app/src/hal/int8_npu.h")
        driver = read_repo("ESP_INT8_app/src/hal/int8_npu.c")
        main = read_repo("ESP_INT8_app/src/main.c")
        for token in (
            "INT8_CONV_WIN_STATE_COUNT",
            "INT8_CONV_SA_STATE_COUNT",
            "INT8_CONV_POST_STATE_COUNT",
            "stage_counter_has_conv_profile",
            "stage_counter_read_conv_win",
            "stage_counter_read_conv_sa",
            "stage_counter_read_conv_post",
        ):
            self.assertIn(token, header + driver)
        self.assertIn("RTL_CONV_CYCLES_BEGIN", driver)
        self.assertIn("RTL_CONV_CHECK", driver)
        self.assertIn("stage_counter_derive_conv_idle", driver)
        self.assertIn('case 0U: return "IDLE_OR_DONE";', driver)
        self.assertIn("conv_win[INT8_CONV_WIN_STATE_COUNT]", main)
        self.assertIn("conv_sa[INT8_CONV_SA_STATE_COUNT]", main)
        self.assertIn("conv_post[INT8_CONV_POST_STATE_COUNT]", main)
        self.assertIn("cw0_idle_done,cw1_active", main)
        self.assertNotIn("cw2_done", main)

    def test_single_image_checks_profiling_and_validates_saved_mask(self) -> None:
        main = read_repo("ESP_INT8_app/src/main.c")
        start = main.index("static int run_single_image(")
        body = main[start:main.index("\nstatic int ", start + 1)]
        self.assertIn("stage_counter_has_conv_profile()", body)
        self.assertIn("validate_output_classes(profile_name, expected_classes)", body)
        self.assertIn("run_full_infer_once(npu, uop_count, &cycles, 1U)", body)
        self.assertIn("s_perf_count = 0U;", body)
        self.assertIn("stage_counter_read_stage(12U)", body)
        self.assertLess(body.index("stage_counter_has_conv_profile()"),
                        body.index("run_full_infer_once("))
        self.assertLess(body.index("SD_SaveMemoryToFile("),
                        body.index("validate_output_classes(profile_name, expected_classes)"))

    def test_dual_single_mode_uses_separate_short_names(self) -> None:
        cfg = read_repo("ESP_INT8_app/src/app_config.h")
        self.assertRegex(cfg, r"#define INT8_APP_ENABLE_DUAL_SINGLE_TEST 1U")
        for mode in ("EXEC_PREFIX_PROFILE", "VAL_SET_TEST", "DUAL_VAL_SET_TEST"):
            self.assertRegex(cfg, rf"#define INT8_APP_ENABLE_{mode} 0U")
        names = []
        for profile in ("BINARY", "CLASS20"):
            for role in ("PARAM", "SINGLE_INPUT", "SINGLE_OUTPUT"):
                match = re.search(rf'#define INT8_APP_{profile}_{role}_FILE "([^"]+)"', cfg)
                self.assertIsNotNone(match, (profile, role))
                filename = match.group(1)
                self.assertRegex(filename, r"^[A-Z0-9_]{1,8}\.BIN$")
                names.append(filename)
        self.assertEqual(len(set(names)), 6)

    def test_dual_single_reinitializes_before_each_inference(self) -> None:
        main = read_repo("ESP_INT8_app/src/main.c")
        self.assertIn("static int run_dual_single_image(", main)
        start = main.index("static int run_dual_single_image(")
        body = main[start:main.index("\nstatic int ", start + 1)]
        self.assertEqual(body.count("load_param_and_init("), 2)
        self.assertEqual(body.count("run_single_image("), 2)
        binary_param = body.index("INT8_APP_BINARY_PARAM_FILE")
        binary_input = body.index("INT8_APP_BINARY_SINGLE_INPUT_FILE")
        class20_param = body.index("INT8_APP_CLASS20_PARAM_FILE")
        class20_input = body.index("INT8_APP_CLASS20_SINGLE_INPUT_FILE")
        self.assertLess(binary_param, binary_input)
        self.assertLess(binary_input, class20_param)
        self.assertLess(class20_param, class20_input)
        self.assertIn("INT8_APP_BINARY_CLASS_COUNT", body)
        self.assertIn("INT8_APP_CLASS20_CLASS_COUNT", body)
        self.assertIn("DUAL_SINGLE_SUMMARY_BEGIN", body)
        self.assertNotIn("run_val_set(", body)
        self.assertNotIn("run_exec_prefix_profile(", body)
        self.assertRegex(main, r"#elif INT8_APP_ENABLE_DUAL_SINGLE_TEST\s+return run_dual_single_image\(&npu\);")

    def test_single_timer_keeps_setup_and_csv_outside_measurement(self) -> None:
        main = read_repo("ESP_INT8_app/src/main.c")
        start = main.index("static int run_infer_once(")
        body = main[start:main.index("\nstatic int ", start + 1)]
        begin = body.index("start = read_timer_counter();")
        end = body.index("end = read_timer_counter();")
        self.assertLess(body.index("stage_counter_clear();"), begin)
        self.assertLess(body.index("Xil_DCacheFlushRange"), begin)
        self.assertLess(begin, body.index("int8_npu_run_infer("))
        self.assertLess(body.index("int8_npu_run_infer("), end)
        self.assertLess(end, body.index("Xil_DCacheInvalidateRange"))
        self.assertLess(end, body.rindex("stage_counter_dump_csv();"))


if __name__ == "__main__":
    unittest.main()
