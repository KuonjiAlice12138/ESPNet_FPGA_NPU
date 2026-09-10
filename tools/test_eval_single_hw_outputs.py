import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

import eval_single_hw_outputs as evaluator


class EvalSingleHwOutputsTests(unittest.TestCase):
    def test_loads_multiclass_contract_from_artifact_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)
            (artifact / "manifest.json").write_text(
                json.dumps(
                    {
                        "deployment": {
                            "profile_name": "cityscapes20",
                            "class_count": 20,
                            "ignore_metric_class": 19,
                        }
                    }
                ),
                encoding="utf-8",
            )

            contract = evaluator.load_deployment_contract(artifact)

        self.assertEqual(contract.profile_name, "cityscapes20")
        self.assertEqual(contract.class_count, 20)
        self.assertEqual(contract.ignore_metric_class, 19)

    def test_multiclass_metrics_exclude_ignore_class_from_valid_pixels(self):
        target = np.array([[0, 1, 19], [2, 2, 19]], dtype=np.int64)
        pred = np.array([[0, 2, 7], [2, 1, 19]], dtype=np.int64)

        metrics = evaluator.segmentation_metrics(
            pred,
            target,
            class_count=20,
            ignore_target=19,
        )

        self.assertEqual(metrics["valid_pixels"], 4)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 0.5)
        self.assertEqual(len(metrics["per_class_IoU"]), 19)
        self.assertEqual([entry["class"] for entry in metrics["confusion"]], list(range(19)))

    def test_multiclass_fixed_upsample_selects_largest_interpolated_logit(self):
        logits = np.zeros((1, 64, 128, 3), dtype=np.int8)
        logits[..., 1] = 3
        logits[..., 2] = 2

        mask = evaluator.fullres_mask_from_logits(logits, class_count=3)

        self.assertEqual(mask.shape, evaluator.FULLRES_SHAPE)
        self.assertTrue(np.all(mask == 1))

    def test_logits_loader_uses_contract_class_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logits.bin"
            np.zeros((1, 64, 128, 20), dtype=np.int8).tofile(path)
            logits = evaluator.load_i8_logits(path, class_count=20)

        self.assertEqual(logits.shape, (1, 64, 128, 20))

    def test_cached_relative_path_resolves_against_espnet_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "city" / "gtFine" / "sample.png"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"target")

            resolved = evaluator.resolve_cached_path(
                "./city//gtFine/sample.png",
                cached_data_file=root / "cache" / "city.p",
                espnet_dir=root,
            )

        self.assertEqual(resolved, target)

    def test_sample_index_matches_absolute_artifact_to_relative_cache_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact"
            artifact.mkdir()
            (artifact / "sample_info.json").write_text(
                json.dumps(
                    {
                        "sample_name": (
                            r"D:\ESPNet\city\leftImg8bit\val\frankfurt\sample.png"
                        )
                    }
                ),
                encoding="utf-8",
            )
            cache = root / "city.p"
            with cache.open("wb") as f:
                pickle.dump(
                    {
                        "valIm": [
                            "./city//leftImg8bit/val/frankfurt/other.png",
                            "./city//leftImg8bit/val/frankfurt/sample.png",
                        ]
                    },
                    f,
                )

            sample_index, sample_name = evaluator.find_sample_index(artifact, cache)

        self.assertEqual(sample_index, 1)
        self.assertEqual(
            sample_name, "./city//leftImg8bit/val/frankfurt/sample.png"
        )


if __name__ == "__main__":
    unittest.main()
