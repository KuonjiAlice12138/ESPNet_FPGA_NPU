import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import export_int8_hw_blob as exporter


class MulticlassDeploymentContractTests(unittest.TestCase):
    def _artifact(self, root: Path, classes: int) -> Path:
        artifact = root / "artifact"
        (artifact / "golden_sample" / "classifier").mkdir(parents=True)
        (artifact / "layers" / "classifier").mkdir(parents=True)
        manifest = {
            "deployment": {
                "profile_name": "binary2" if classes == 2 else "cityscapes20",
                "class_count": classes,
                "output_semantics": "bilinear_logits_then_argmax",
                "classifier_activation_zero_point": 0,
            },
            "classifier": {"weight_shape": [classes, 256, 1, 1]},
        }
        (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        np.save(
            artifact / "golden_sample" / "classifier" / "output_int.npy",
            np.zeros((1, classes, 64, 128), dtype=np.int8),
        )
        np.save(
            artifact / "layers" / "classifier" / "weight_int8.npy",
            np.zeros((classes, 256, 1, 1), dtype=np.int8),
        )
        return artifact

    def test_contract_drives_classifier_tensor_conv_and_uops(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = exporter.load_deployment_contract(self._artifact(Path(tmp), 20))
        tensors, convs = exporter.build_deployment_plans(contract.class_count)
        uops = exporter.build_uops(contract.class_count)
        self.assertEqual(tensors[17].c, 20)
        self.assertEqual(convs[25].out_c, 20)
        self.assertEqual(uops[-3].out_c, 20)
        self.assertEqual(uops[-2].valid_c, 20)
        self.assertEqual(len(uops), 75)

    def test_manifest_shape_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = self._artifact(Path(tmp), 20)
            np.save(
                artifact / "golden_sample" / "classifier" / "output_int.npy",
                np.zeros((1, 2, 64, 128), dtype=np.int8),
            )
            with self.assertRaisesRegex(ValueError, "golden classifier"):
                exporter.load_deployment_contract(artifact)

    def test_binary_plan_is_restored_after_multiclass_plan(self):
        exporter.activate_deployment_plans(20)
        self.assertEqual(exporter.CONV_PLAN[25].out_c, 20)
        exporter.activate_deployment_plans(2)
        self.assertEqual(exporter.CONV_PLAN[25].out_c, 2)
        self.assertEqual(exporter.TENSOR_DESC_BY_ID[17].c, 2)


if __name__ == "__main__":
    unittest.main()
