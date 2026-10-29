import hashlib
import json
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
APP_YAML = (ROOT / "ESP_INT8_app" / "src" / "app.yaml").read_text(
    encoding="utf-8"
)
README = (ROOT / "ESP_INT8_app" / "src" / "README.txt").read_text(
    encoding="utf-8"
)


def macro(name: str) -> str:
    match = re.search(rf"^#define\s+{re.escape(name)}\s+(.+?)\s*$", CONFIG, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing macro {name}")
    return match.group(1)


class GraySingleAppContract(unittest.TestCase):
    def test_fixed_h240w240_c1_geometry(self):
        self.assertEqual(macro("INT8_APP_INPUT_H"), "240U")
        self.assertEqual(macro("INT8_APP_INPUT_W"), "240U")
        self.assertEqual(macro("INT8_APP_INPUT_C"), "1U")
        self.assertEqual(macro("INT8_APP_OUTPUT_H"), "240U")
        self.assertEqual(macro("INT8_APP_OUTPUT_W"), "240U")
        self.assertIn("INT8_APP_INPUT_H * INT8_APP_INPUT_W * INT8_APP_INPUT_C",
                      macro("INT8_INPUT_BYTES"))
        self.assertIn("INT8_APP_OUTPUT_H * INT8_APP_OUTPUT_W",
                      macro("INT8_OUTPUT_BYTES"))

    def test_binary_single_full_network_mode_is_exclusive(self):
        self.assertEqual(macro("INT8_APP_ENABLE_INPUT_SOURCE_TEST"), "1U")
        self.assertEqual(macro("INT8_APP_ENABLE_DUAL_SINGLE_TEST"), "0U")
        self.assertEqual(macro("INT8_APP_ENABLE_EXEC_PREFIX_PROFILE"), "0U")
        self.assertEqual(macro("INT8_APP_ENABLE_VAL_SET_TEST"), "0U")
        self.assertEqual(macro("INT8_APP_ENABLE_DUAL_VAL_SET_TEST"), "0U")
        self.assertEqual(macro("INT8_APP_DEPLOYMENT_NAME"), '"binary2"')
        self.assertEqual(macro("INT8_APP_EXPECTED_CLASS_COUNT"), "2U")
        self.assertIn("return run_input_source_tests(&npu, header.uop_count)", MAIN)

    def test_new_platform_requires_conv_internal_profile(self):
        self.assertEqual(macro("INT8_APP_REQUIRE_CONV_PROFILE"), "1U")
        self.assertIn("stage_counter_dump_conv_csv();", HAL)
        self.assertIn("stage_counter_has_conv_profile()", MAIN)

    def test_round1_post_profile_uses_word_class_labels(self):
        start = HAL.index("static const char *stage_counter_conv_post_name(")
        end = HAL.index("\nstatic void ", start + 1)
        names = HAL[start:end]
        self.assertIn('case 1U: return "STAGING_WORD_OR_WAIT_PSUM";', names)
        self.assertIn('case 2U: return "COMMIT_WORD_OR_WAIT_PSUM";', names)
        self.assertNotIn('return "ROWBUF_WRITE";', names)
        self.assertNotIn('return "REQUANT_OR_WAIT_PSUM";', names)
        self.assertIn("POST bins classify psum words", HAL)
        self.assertIn("not phase times", HAL)

    def test_build_identifies_h240_platform_and_retains_timer_units(self):
        self.assertEqual(macro("INT8_APP_BUILD_TAG"),
                         '"INT8-BOARD-20261025-P7-H240-RESIZE-H512-B2"')
        self.assertIn(r"platform_h240_resize_1025\export\platform_h240_resize_1025", APP_YAML)
        self.assertIn("Platform: platform_h240_resize_1025", README)
        self.assertIn("WNS=+0.667 ns", README)
        self.assertIn("WHS=+0.010 ns", README)
        self.assertIn("routing errors=0", README)
        self.assertEqual(macro("INT8_APP_PL_TARGET_HZ"), "100000000ULL")
        self.assertEqual(macro("INT8_APP_PL_TARGET_PERIOD_NS"), "10U")
        self.assertIn("CNTPCT_EL0", MAIN)
        self.assertIn("CNTFRQ_EL0", MAIN)
        launch_path = ROOT / "ESP_INT8_app" / "_ide" / "launch.json"
        if launch_path.exists():
            launch = launch_path.read_text(encoding="utf-8")
            self.assertIn("platform_h240_resize_1025", launch)
            for retired in ("platform_h240_1021", "platform_1014", "platform_h240_1012", "platform_h240_1019",
                            "platform_h240_1020", "platform_r5a_0928"):
                self.assertNotIn(retired, launch)
            configs = json.loads(launch)["configurations"]
            for config in configs:
                setup = config["targetSetup"]
                self.assertIn("platform_h240_resize_1025", setup["bitstreamFile"])
                init = setup["zuInitialization"]
                # Vitis keeps the inactive initialization alternative in launch.json.
                if init["isFsbl"]:
                    self.assertTrue(init["usingFSBL"]["initWithFSBL"])
                    self.assertIn("platform_h240_resize_1025", init["usingFSBL"]["fsblFile"])
                else:
                    self.assertTrue(init["usingPsuInit"]["runPsuInit"])
                    self.assertIn("platform_h240_resize_1025", init["usingPsuInit"]["psuInitTclFile"])
        else:
            self.assertIn("Create a new Run/Debug configuration", README)

    def test_boot_recipe_and_component_use_the_same_platform(self):
        component = json.loads(
            (ROOT / "ESP_INT8_app" / "vitis-comp.json").read_text(encoding="utf-8")
        )
        self.assertEqual(component["platform"], "platform_h240_resize_1025")
        self.assertEqual(component["domain"], "standalone_psu_cortexa53_0")
        bif = (ROOT / "ESP_INT8_app" / "src" / "boot_h240.bif").read_text(
            encoding="utf-8"
        )
        self.assertIn("platform_h240_resize_1025/export/platform_h240_resize_1025/sw/boot/fsbl.elf", bif)
        self.assertIn("platform_h240_resize_1025/hw/sdt/ESP_INT8_hwplat.bit", bif)
        self.assertIn("ESP_INT8_app/build/ESP_INT8_app.elf", bif)
        self.assertNotIn("platform_h240_1012", bif)
        self.assertNotIn("platform_h240_1019", bif)
        self.assertNotIn("platform_h240_1020", bif)

    def test_release_selects_round6_transfer_protocol_artifact(self):
        artifact = ROOT / "hw_artifacts" / "binary2_int8_h240w240_gray_r6_v4"
        manifest = json.loads(
            (artifact / "export_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["hardware_profile"], "p7_h240w240_gray_c1_r6_delta")
        self.assertEqual(manifest["param_version"], 4)
        self.assertIn("hw_artifacts/binary2_int8_h240w240_gray_r6_v4", README)
        for filename, size in (("PARAM.BIN", 143040), ("INPUTQ.BIN", 57600),
                               ("golden_fullres_mask.bin", 57600)):
            payload = (artifact / filename).read_bytes()
            self.assertEqual(len(payload), size)
            self.assertIn(hashlib.sha256(payload).hexdigest(), README)

    def test_model_files_are_short_and_resolution_specific(self):
        expected = {
            "INT8_PARAM_FILE": '"P2G.BIN"',
            "INT8_INPUT_FILE": '"I2G.BIN"',
            "INT8_OUTPUT_FILE": '"O2G.BIN"',
        }
        for name, value in expected.items():
            self.assertEqual(macro(name), value)
            self.assertLessEqual(len(value.strip('"').split(".")[0]), 8)

    def test_full_param_v4_contract_is_retained(self):
        self.assertEqual(macro("INT8_PARAM_BLOB_VERSION_CURRENT"), "4U")
        self.assertEqual(macro("INT8_EXPECTED_UOP_COUNT"), "75U")
        self.assertEqual(macro("INT8_EXPECTED_EXEC_PLAN_COUNT"), "16U")
        self.assertIn("incomplete exec plan", MAIN)
        init_start = MAIN.index("static int load_param_and_init(")
        init_end = MAIN.index("\nstatic int ", init_start + 1)
        init_body = MAIN[init_start:init_end]
        self.assertNotIn("#if INT8_APP_ENABLE_DUAL_SINGLE_TEST", init_body)
        main_body = MAIN[MAIN.index("int main(void)"):]
        self.assertIn("load_param_and_init(&npu, INT8_PARAM_FILE, &header)", main_body)
        self.assertNotIn("SD_LoadFileToMemory(INT8_PARAM_FILE", main_body)

    def test_single_model_gets_full_stage_dump_and_error_check(self):
        self.assertIn("run_full_infer_once(npu, uop_count, src_mode, &cycles, 1U)", MAIN)
        self.assertIn("stage_counter_read_stage(12U) != 0ULL", MAIN)

    def test_timing_excludes_cache_sd_and_counter_printing(self):
        start = MAIN.index("static int run_infer_once(")
        end = MAIN.index("\nstatic int ", start + 1)
        body = MAIN[start:end]
        self.assertLess(body.index("Xil_DCacheFlushRange"),
                        body.index("start = read_timer_counter()"))
        self.assertLess(body.index("end = read_timer_counter()"),
                        body.index("Xil_DCacheInvalidateRange"))
        self.assertNotIn("SD_LoadFileToMemory", body)
        self.assertNotIn("SD_SaveMemoryToFile", body)
        self.assertIn("INT8_NPU_TIMEOUT_POLLS", body)
        single_start = MAIN.index("static int run_single_image(")
        single_end = MAIN.index("\nstatic int ", single_start + 1)
        single_body = MAIN[single_start:single_end]
        self.assertLess(single_body.index("*arm_ticks = cycles"),
                        single_body.index("validate_output_classes("))

    def test_all_ddr_buffers_are_cache_line_aligned(self):
        for name in ("g_input", "g_output", "g_param"):
            self.assertRegex(MAIN, rf"{name}\[.*?\].*?aligned\(64\)")
        self.assertEqual(macro("INT8_PARAM_HW_WORDS"), "8192U")
        self.assertEqual(macro("INT8_PARAM_HW_WORD_BYTES"), "32U")

    def test_source_capacity_is_not_the_encoder_geometry(self):
        self.assertEqual(macro("INT8_INPUT_CAPACITY_BYTES"), "524288U")
        self.assertIn("g_input[INT8_INPUT_CAPACITY_BYTES]", MAIN)
        self.assertEqual(macro("INT8_INPUT_SRC_NATIVE"), "0U")
        self.assertEqual(macro("INT8_INPUT_SRC_H256W512"), "1U")
        self.assertEqual(macro("INT8_INPUT_SRC_H512W1024"), "2U")
        self.assertIn("int8_npu_source_bytes(src_mode)", MAIN)
        self.assertIn("Xil_DCacheFlushRange((UINTPTR)g_input, input_bytes)", MAIN)
        self.assertIn("input_size != input_bytes", MAIN)

    def test_generated_source_driver_is_required_and_mode_is_written_before_start(self):
        self.assertIn("XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_SRC_MODE_DATA", HAL)
        self.assertRegex(HAL, r'#error "[^"]*src_mode[^"]*"')
        self.assertIn("XEspnet_encoder_int8_core_Set_src_mode", HAL)
        self.assertIn("XEspnet_encoder_int8_core_Get_src_mode", HAL)
        start = HAL.index("static int run_once(")
        body = HAL[start:HAL.index("\nint int8_npu_run_init(", start)]
        self.assertLess(body.index("XEspnet_encoder_int8_core_Set_src_mode"),
                        body.index("XEspnet_encoder_int8_core_Start"))
        self.assertIn("int8_npu_source_bytes(src_mode) == 0U", body)
        self.assertNotRegex(body, r"(?:SRC_MODE_DATA|src_mode).*0x[0-9a-fA-F]{2}")

    def test_one_boot_selects_only_h512_with_distinct_short_outputs(self):
        self.assertEqual(macro("INT8_APP_SOURCE_TEST_MODE"), "INT8_INPUT_SRC_H512W1024")
        for name, value in (("INT8_APP_SOURCE256_INPUT_FILE", '"I256Q.BIN"'),
                            ("INT8_APP_SOURCE512_INPUT_FILE", '"I512Q.BIN"'),
                            ("INT8_APP_SOURCE256_OUTPUT_FILE", '"O256Q.BIN"'),
                            ("INT8_APP_SOURCE512_OUTPUT_FILE", '"O512Q.BIN"')):
            self.assertEqual(macro(name), value)
        start = MAIN.index("static int run_input_source_tests(")
        body = MAIN[start:MAIN.index("\nstatic int ", start + 1)]
        for name in ("INT8_INPUT_SRC_NATIVE", "INT8_INPUT_SRC_H256W512",
                     "INT8_INPUT_SRC_H512W1024", "INPUT_SOURCE_SUMMARY_BEGIN"):
            self.assertIn(name, body)
        self.assertEqual(body.count("cases[index].mode != INT8_APP_SOURCE_TEST_MODE"), 2,
                         "Only the selected mode may run or appear in the summary")
        self.assertNotIn("load_param_and_init(", body)


if __name__ == "__main__":
    unittest.main()
