import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

import export_int8_hw_blob as exporter
from geometry_contract import DEFAULT_GEOMETRY, DeploymentGeometry
import eval_val_hw_masks_fullres as eval_fullres
import eval_single_hw_outputs as evaluator


class Round1GeometryContractTests(unittest.TestCase):
    def test_default_geometry_is_h256w512(self):
        self.assertEqual(DEFAULT_GEOMETRY.input_height, 256)
        self.assertEqual(DEFAULT_GEOMETRY.input_width, 512)
        self.assertEqual(DEFAULT_GEOMETRY.logits_height, 32)
        self.assertEqual(DEFAULT_GEOMETRY.logits_width, 64)
        self.assertEqual(DEFAULT_GEOMETRY.to_manifest()["target_scale"], 8)

    def test_compiler_plan_is_geometry_and_class_driven(self):
        geometry = DeploymentGeometry(256, 512, 8)
        tensors, convs = exporter.build_deployment_plans(20, geometry=geometry)
        uops = exporter.build_uops(20, geometry=geometry)

        self.assertEqual((tensors[0].h, tensors[0].w), (256, 512))
        self.assertEqual((tensors[17].h, tensors[17].w, tensors[17].c), (32, 64, 20))
        self.assertEqual((tensors[18].h, tensors[18].w), (128, 256))
        self.assertEqual(convs[25].out_c, 20)
        self.assertEqual((uops[-3].in_h, uops[-3].in_w, uops[-3].out_c), (32, 64, 20))
        self.assertEqual((uops[-2].in_h, uops[-2].in_w, uops[-2].valid_c), (32, 64, 20))
        self.assertEqual(len(uops), 75)

    def test_eval_contract_exposes_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)
            (artifact / "manifest.json").write_text(
                json.dumps(
                    {
                        "geometry": DEFAULT_GEOMETRY.to_manifest(),
                        "deployment": {
                            "profile_name": "cityscapes20",
                            "class_count": 20,
                            "ignore_metric_class": 19,
                        },
                    }
                ),
                encoding="utf-8",
            )
            contract = evaluator.load_deployment_contract(artifact)

        self.assertEqual(contract.geometry, DEFAULT_GEOMETRY)
        self.assertEqual(contract.lowres_shape, (32, 64))
        self.assertEqual(contract.fullres_shape, (256, 512))

    def test_board_eval_geometry_comes_from_manifest(self):
        geometry = eval_fullres.load_eval_geometry(
            sd_manifest={"geometry": DEFAULT_GEOMETRY.to_manifest()},
            param_audit=None,
        )
        self.assertEqual(geometry, DEFAULT_GEOMETRY)
        self.assertEqual(geometry.output_mask_bytes, 256 * 512)

    def test_exporter_wbuf_matches_hls_capacity(self):
        config = (
            Path(__file__).resolve().parents[1]
            / "ESP_INT8_hls"
            / "include"
            / "npu_config.hpp"
        ).read_text(encoding="utf-8")
        match = re.search(r"constexpr int WBUF_BYTES = (\d+)\s*\*\s*1024;", config)
        self.assertIsNotNone(match)
        self.assertEqual(exporter.WBUF_BYTES, int(match.group(1)) * 1024)
        self.assertGreaterEqual(exporter.WBUF_BYTES, 123456)

    def test_hardware_golden_preserves_model_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            model = np.array([[[[1, -2], [3, -4]]]], dtype=np.int8)
            replay = np.array([[[1, -1], [3, -5]]], dtype=np.int8)
            np.save(out_dir / "golden_output_q_nhwc.npy", model)
            model.tofile(out_dir / "golden_output_q.bin")
            model.tofile(out_dir / "expected_output_q.bin")

            report = exporter.write_hardware_golden_files(out_dir, replay)

            np.testing.assert_array_equal(
                np.load(out_dir / "model_golden_output_q_nhwc.npy"), model
            )
            np.testing.assert_array_equal(
                np.load(out_dir / "golden_output_q_nhwc.npy"), replay[np.newaxis, ...]
            )
            self.assertEqual(
                np.fromfile(out_dir / "model_golden_output_q.bin", dtype=np.int8).tolist(),
                model.reshape(-1).tolist(),
            )
            self.assertEqual(
                np.fromfile(out_dir / "golden_output_q.bin", dtype=np.int8).tolist(),
                replay.reshape(-1).tolist(),
            )
            self.assertEqual(
                np.fromfile(out_dir / "expected_output_q.bin", dtype=np.int8).tolist(),
                replay.reshape(-1).tolist(),
            )
            self.assertEqual(report["model_vs_hardware"]["mismatches"], 2)
            self.assertEqual(report["golden_semantics"], "param_v4_integer_replay")


if __name__ == "__main__":
    unittest.main()
