import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HLS = ROOT / "ESP_INT8_hls"


def read(relative: str) -> str:
    return (HLS / relative).read_text(encoding="utf-8")


def constexpr_int(source: str, name: str) -> int:
    match = re.search(rf"constexpr\s+int\s+{name}\s*=\s*(\d+)\s*;", source)
    if match is None:
        raise AssertionError(f"missing literal constexpr {name}")
    return int(match.group(1))


class Round2HlsGeometryContractTest(unittest.TestCase):
    def test_fixed_geometry_and_axi_depths(self) -> None:
        config = read("include/npu_config.hpp")
        self.assertEqual(constexpr_int(config, "INPUT_FRAME_H"), 256)
        self.assertEqual(constexpr_int(config, "INPUT_FRAME_W"), 512)
        self.assertEqual(constexpr_int(config, "ENCODER_OUT_H"), 32)
        self.assertEqual(constexpr_int(config, "ENCODER_OUT_W"), 64)

        top = read("src/int8_core.cpp")
        self.assertIn("INPUT_FRAME_AXI_WORDS == 12288", top)
        self.assertIn("OUTPUT_FRAME_AXI_WORDS == 4096", top)
        self.assertRegex(top, r"port=gmem_frame_in\s+depth=12288")
        self.assertRegex(top, r"port=gmem_frame_out\s+depth=4096")

    def test_param_init_rejects_wrong_deployment_geometry(self) -> None:
        param_dma = read("src/param_dma.cpp")
        self.assertIn("validate_deployment_geometry", param_dma)
        self.assertIn("TID_INPUT", param_dma)
        self.assertIn("TID_OUT", param_dma)
        self.assertIn("INPUT_FRAME_H", param_dma)
        self.assertIn("ENCODER_OUT_H", param_dma)

    def test_fullres_tb_uses_round1_artifacts(self) -> None:
        top_tb = read("tb/top_golden_sample_tb.cpp")
        self.assertIn("cityscapes20_int8_h256w512_v4", top_tb)
        self.assertIn("binary2_int8_h256w512_v4", top_tb)
        self.assertEqual(top_tb.count("_int8_0809"), 1)
        self.assertIn("legacy H512W1024 PARAM passed H256W512 geometry gate", top_tb)

    def test_upsample_geometry_is_compile_time_consistent(self) -> None:
        config = read("include/npu_config.hpp")
        self.assertIn("INPUT_FRAME_H == ENCODER_OUT_H * UPSAMPLE_SCALE", config)
        self.assertIn("INPUT_FRAME_W == ENCODER_OUT_W * UPSAMPLE_SCALE", config)


if __name__ == "__main__":
    unittest.main()
