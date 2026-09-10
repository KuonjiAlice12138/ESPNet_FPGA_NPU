import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "ESP_INT8_app" / "src" / "app_config.h").read_text(
    encoding="utf-8"
)
MAIN = (ROOT / "ESP_INT8_app" / "src" / "main.c").read_text(encoding="utf-8")
HAL = (ROOT / "ESP_INT8_app" / "src" / "hal" / "int8_npu.c").read_text(
    encoding="utf-8"
)


def macro(name: str) -> str:
    match = re.search(rf"^#define\s+{re.escape(name)}\s+(.+?)\s*$", CONFIG, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing macro {name}")
    return match.group(1)


class DualProfileAppContract(unittest.TestCase):
    def test_fixed_h256w512_geometry(self):
        self.assertEqual(macro("INT8_APP_INPUT_H"), "256U")
        self.assertEqual(macro("INT8_APP_INPUT_W"), "512U")
        self.assertEqual(macro("INT8_APP_INPUT_C"), "3U")
        self.assertEqual(macro("INT8_APP_OUTPUT_H"), "256U")
        self.assertEqual(macro("INT8_APP_OUTPUT_W"), "512U")
        self.assertIn("INT8_APP_INPUT_H * INT8_APP_INPUT_W * INT8_APP_INPUT_C",
                      macro("INT8_INPUT_BYTES"))
        self.assertIn("INT8_APP_OUTPUT_H * INT8_APP_OUTPUT_W",
                      macro("INT8_OUTPUT_BYTES"))

    def test_dual_single_full_network_mode_is_exclusive(self):
        self.assertEqual(macro("INT8_APP_ENABLE_DUAL_SINGLE_TEST"), "1U")
        self.assertEqual(macro("INT8_APP_ENABLE_EXEC_PREFIX_PROFILE"), "0U")
        self.assertEqual(macro("INT8_APP_ENABLE_VAL_SET_TEST"), "0U")
        self.assertEqual(macro("INT8_APP_ENABLE_DUAL_VAL_SET_TEST"), "0U")
        self.assertIn("return run_dual_single_image(&npu);", MAIN)

    def test_new_platform_requires_conv_internal_profile(self):
        self.assertEqual(macro("INT8_APP_REQUIRE_CONV_PROFILE"), "1U")
        self.assertIn("stage_counter_dump_conv_csv();", HAL)
        self.assertIn("stage_counter_has_conv_profile()", MAIN)

    def test_model_files_are_short_and_resolution_specific(self):
        expected = {
            "INT8_APP_BINARY_PARAM_FILE": '"P2H.BIN"',
            "INT8_APP_BINARY_SINGLE_INPUT_FILE": '"I2H.BIN"',
            "INT8_APP_BINARY_SINGLE_OUTPUT_FILE": '"O2H.BIN"',
            "INT8_APP_CLASS20_PARAM_FILE": '"P20H.BIN"',
            "INT8_APP_CLASS20_SINGLE_INPUT_FILE": '"I20H.BIN"',
            "INT8_APP_CLASS20_SINGLE_OUTPUT_FILE": '"O20H.BIN"',
        }
        for name, value in expected.items():
            self.assertEqual(macro(name), value)
            self.assertLessEqual(len(value.strip('"').split(".")[0]), 8)

    def test_full_param_v4_contract_is_retained(self):
        self.assertEqual(macro("INT8_PARAM_BLOB_VERSION_CURRENT"), "4U")
        self.assertEqual(macro("INT8_EXPECTED_UOP_COUNT"), "75U")
        self.assertEqual(macro("INT8_EXPECTED_EXEC_PLAN_COUNT"), "16U")
        self.assertIn("incomplete exec plan", MAIN)

    def test_each_model_gets_full_stage_dump_and_error_check(self):
        self.assertIn("run_full_infer_once(npu, uop_count, &cycles, 1U)", MAIN)
        self.assertIn("stage_counter_read_stage(12U) != 0ULL", MAIN)
        self.assertRegex(
            MAIN, r'run_single_image\(\s*npu, header\.uop_count, "binary2"'
        )
        self.assertRegex(
            MAIN, r'run_single_image\(\s*npu, header\.uop_count, "cityscapes20"'
        )

    def test_one_model_failure_does_not_skip_the_other_profile(self):
        start = MAIN.index("static int run_dual_single_image(")
        end = MAIN.index("\nstatic int ", start + 1)
        body = MAIN[start:end]
        self.assertIn("binary_status", body)
        self.assertIn("class20_status", body)
        self.assertIn("profile,classes,result,arm_ticks", body)
        self.assertEqual(body.count("return "), 1)

        single_start = MAIN.index("static int run_single_image(")
        single_end = MAIN.index("\nstatic int ", single_start + 1)
        single_body = MAIN[single_start:single_end]
        self.assertLess(single_body.index("*arm_ticks = cycles"),
                        single_body.index("validate_output_classes("))


if __name__ == "__main__":
    unittest.main()
