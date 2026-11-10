import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import eval_single_hw_outputs as evaluator
import eval_val_hw_masks_fullres as val_evaluator
from geometry_contract import DeploymentGeometry


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


class HighResolutionMaskTests(unittest.TestCase):
    geometry = DeploymentGeometry(240, 240, 8, input_channels=1)

    def test_exact_512_axis_and_clipped_borders(self):
        maps = [evaluator.bilinear_axis_map_to_size(d, 30, 512) for d in range(512)]
        self.assertEqual(maps[:9], [(0, 0, 512, 0)] * 9)
        self.assertEqual(maps[9], (0, 1, 483, 29))
        self.assertEqual(maps[-9:], [(29, 29, 512, 0)] * 9)
        for d, (i0, i1, w0, w1) in enumerate(maps):
            self.assertEqual(w0 + w1, 512)
            if i0 != i1:
                self.assertEqual(i0 * 512 + w1, 30 * d - 241)

    def test_explicit_240_shape_preserves_the_old_mask(self):
        logits = np.random.default_rng(1104).integers(-128, 128, (30, 30, 2), dtype=np.int8)
        old = evaluator.fullres_mask_from_logits(logits, geometry=self.geometry)
        new = evaluator.fullres_mask_from_logits(logits, geometry=self.geometry, mask_shape=(240, 240))
        np.testing.assert_array_equal(new, old)

    def test_512_interpolates_logits_before_argmax(self):
        import torch
        import torch.nn.functional as functional
        logits = np.random.default_rng(1105).integers(-128, 128, (30, 30, 2), dtype=np.int8)
        got = evaluator.fullres_mask_from_logits(logits, geometry=self.geometry, mask_shape=(512, 512))
        reference = functional.interpolate(
            torch.from_numpy(logits.transpose(2, 0, 1).copy())[None].double(),
            size=(512, 512), mode="bilinear", align_corners=False,
        ).argmax(1)[0].numpy()
        self.assertEqual(got.shape, (512, 512))
        self.assertEqual(got.dtype, np.uint8)
        np.testing.assert_array_equal(got, reference)

    def test_ties_and_signed_extremes_keep_the_class_zero_rule(self):
        logits = np.full((30, 30, 2), -128, dtype=np.int8)
        self.assertFalse(evaluator.fullres_mask_from_logits(
            logits, geometry=self.geometry, mask_shape=(512, 512)).any())
        logits[:, :, 1] = 127
        self.assertTrue(np.all(evaluator.fullres_mask_from_logits(
            logits, geometry=self.geometry, mask_shape=(512, 512)) == 1))

    def test_invalid_mask_shapes_are_rejected(self):
        logits = np.zeros((30, 30, 2), np.int8)
        for shape in ((0, 512), (512,), (512.0, 512), (512, -1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                evaluator.fullres_mask_from_logits(logits, geometry=self.geometry, mask_shape=shape)

    def test_manifest_keeps_encoder_and_mask_shapes_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"deployment": {"profile_name": "binary2", "class_count": 2},
                        "geometry": self.geometry.to_manifest(), "mask_shape": [512, 512],
                        "target_encoding": "cityscapes_train_ids",
                        "preprocessing": {"center_square_crop": True}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            contract = evaluator.load_deployment_contract(root)
        self.assertEqual(contract.fullres_shape, (512, 512))
        self.assertEqual(contract.lowres_shape, (30, 30))
        self.assertEqual(contract.geometry.input_shape_nhwc, (240, 240, 1))
        self.assertEqual(val_evaluator.load_eval_mask_shape(manifest, None, self.geometry), (512, 512))

    def test_lowres_target_does_not_take_a_stride_from_the_512_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.npy"
            target = np.indices((240, 240))[1].astype(np.uint8) % 2
            np.save(path, target[None])
            contract = evaluator.DeploymentContract(
                "binary2", 2, None, self.geometry, mask_shape=(512, 512))
            np.testing.assert_array_equal(evaluator.load_lowres_target(path, contract), target[::8, ::8])

    def test_raw_train_ids_are_not_inferred_as_binary_ids(self):
        raw_contract = evaluator.DeploymentContract(
            "binary2", 2, None, self.geometry, target_encoding="cityscapes_train_ids")
        raw = np.array([[0, 1, 13, 18, 255]], np.uint8)
        np.testing.assert_array_equal(evaluator.remap_target_np(raw, raw_contract), [[1, 1, 0, 0, 255]])
        np.testing.assert_array_equal(evaluator.remap_target_np(raw[:, :2], raw_contract), [[1, 1]])
        mapped_contract = evaluator.DeploymentContract(
            "binary2", 2, None, self.geometry, target_encoding="binary2_ids")
        np.testing.assert_array_equal(evaluator.remap_target_np(raw[:, :2], mapped_contract), [[0, 1]])
        with self.assertRaises(ValueError):
            evaluator.remap_target_np(raw, mapped_contract)

    def test_raw_annotation_encoding_does_not_remap_the_mapped_npy_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.npy"
            target = np.indices((240, 240))[1].astype(np.uint8) % 2
            np.save(path, target[None])
            contract = evaluator.DeploymentContract(
                "binary2", 2, None, self.geometry, mask_shape=(512, 512),
                target_encoding="cityscapes_train_ids")
            np.testing.assert_array_equal(evaluator.load_lowres_target(path, contract), target[::8, ::8])

    def test_highres_targets_come_from_the_raw_center_not_a_240_label(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = np.zeros((1024, 2048), np.uint8)
            labels[1::2, 513:1536:2] = 13
            labels[1, 513] = 255
            image_path = root / "sample_leftImg8bit.png"
            target_path = root / "sample_gtFine_labelTrainIds.png"
            cv2.imwrite(str(target_path), labels)
            cache_path = root / "cache.pkl"
            cache_path.write_bytes(pickle.dumps({"valIm": [str(image_path)], "valAnnot": [str(target_path)]}))
            dataset = val_evaluator.build_fullres_val_dataset(
                root, cache_path, self.geometry, {"center_square_crop": True}, "binary2",
                mask_shape=(512, 512), target_encoding="cityscapes_train_ids")
            _, target, _ = dataset[0]
            self.assertEqual(tuple(target.shape), (512, 512))
            self.assertEqual(int(target[0, 0]), 255)
            self.assertTrue(np.all(target.numpy().ravel()[1:] == 0))

    def test_board_mask_loader_uses_mask_bytes_not_encoder_input_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "O0000.BIN"
            np.ones((512, 512), np.uint8).tofile(path)
            mask = val_evaluator.load_mask(path, self.geometry, mask_shape=(512, 512))
            self.assertEqual(mask.shape, (512, 512))
            with self.assertRaises(ValueError):
                val_evaluator.load_mask(path, self.geometry)


class ParamReplayIntegerTests(unittest.TestCase):
    def test_non_bypass_add_rounds_signed_halves_and_saturates_without_tensor_helpers(self):
        import hw_param_replay as replay
        a = np.array([-128, -5, -1, 0, 1, 5, 127], np.int8).reshape(1, -1, 1)
        b = np.array([-128, 2, 0, 0, 0, 2, 127], np.int8).reshape(1, -1, 1)
        q = SimpleNamespace(requant_bypass=False, mult=1, shift=1, act_type=0)
        np.testing.assert_array_equal(replay.add_i8_array(a, b, q).ravel(), [-128, -2, -1, 0, 1, 4, 127])
        q.act_type = replay.ACT_RELU
        np.testing.assert_array_equal(replay.add_i8_array(a, b, q).ravel(), [0, 0, 0, 0, 1, 4, 127])


if __name__ == "__main__":
    unittest.main()
