"""Round3 evaluation/dataset contracts; tiny fixtures, no training or HLS build."""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np

import eval_single_hw_outputs as evaluator
import export_int8_hw_blob as compiler
import export_val_hw_dataset as dataset
import geometry_contract as geometry


ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    0: ROOT / "hw_artifacts/binary2_int8_h240w240_gray_r6_v4",
    3: ROOT / "hw_artifacts/cityscapes20_int8_h256w512_r2_v4",
    4: ROOT / "hw_artifacts/binary2_int8_h512w512_gray_finetuned_v4",
}


def identity(path, semantics=None):
    data = path.read_bytes()
    result = dict(file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    if semantics is not None:
        result.update(crc32=f"{zlib.crc32(data) & 0xffffffff:08X}", semantics=semantics)
    return result


def write_manifest(directory, manifest):
    (directory / "MODEL.JSN").write_text(json.dumps(manifest), encoding="utf-8")


def artifact(directory, selection, *, target=True, frozen_logits=False):
    directory.mkdir()
    old = FROZEN[selection.profile.profile_id]
    raw = (old / "PARAM.BIN").read_bytes()
    (directory / "PARAM.BIN").write_bytes(compiler.recompile_runtime_param(raw, selection))
    source = np.arange(selection.input_bytes, dtype=np.int64).astype(np.int8)
    if frozen_logits:
        source = np.fromfile(old / "input_q.bin", dtype=np.int8)
    source = source.reshape(selection.source_shape)
    source.tofile(directory / "INPUTQ.BIN")
    geometry.select_runtime_input(source, selection).tofile(directory / "ENCODER.BIN")
    g = selection.profile.geometry
    logits = np.zeros((g.logits_height, g.logits_width, selection.classes), dtype=np.int8)
    logits[..., 0] = 2
    logits[:, g.logits_width // 2:, 1] = 7
    if selection.classes == 20:
        logits[g.logits_height // 2:, :, 7] = 12
    if frozen_logits:
        logits = np.fromfile(old / "golden_output_q.bin", dtype=np.int8).reshape(logits.shape)
    logits.tofile(directory / "LOGITS.BIN")
    mask = geometry.runtime_mask_from_logits(logits, selection)
    mask.tofile(directory / "MASK.BIN")
    files = {
        "param": ("PARAM.BIN", "PARAM_v4_unified"),
        "input": ("INPUTQ.BIN", "source_HWC_int8_prequantized"),
        "encoder": ("ENCODER.BIN", "encoder_HWC_int8_selected"),
        "logits": ("LOGITS.BIN", "logits_HWC_int8"),
        "mask": ("MASK.BIN", "mask_HW_uint8_logits_then_argmax"),
    }
    if target:
        gt = mask.copy()
        gt[0, 0] = 255
        gt.tofile(directory / "TARGET.BIN")
        files["target"] = ("TARGET.BIN", "mapped_target_HW_uint8")
    manifest = {
        "format": "P7_UNIFIED_INT8_ARTIFACT_V1",
        "runtime": selection.to_manifest(),
        "geometry": g.to_manifest(),
        "deployment": {
            "profile_name": "binary2" if selection.classes == 2 else "cityscapes20",
            "class_count": selection.classes,
            "ignore_metric_class": None if selection.classes == 2 else 19,
        },
        "preprocessing": {"image_mode": "gray" if g.input_channels == 1 else "rgb",
                          "center_square_crop": g.input_channels == 1},
        "target": {"available": target},
        "sample": {"sample_name": "tiny-validation-sample"},
        "provenance": {
            "numeric_sha256": hashlib.sha256(
                raw[struct.unpack_from("<32I", raw)[17]:struct.unpack_from("<32I", raw)[16]]
            ).hexdigest(),
            "frozen_logits_sha256": evaluator.sha256_file(directory / "LOGITS.BIN"),
        },
        "files": {key: identity(directory / name, semantics)
                  for key, (name, semantics) in files.items()},
    }
    write_manifest(directory, manifest)
    return manifest


def eval_args(directory, **kwargs):
    return argparse.Namespace(artifact_dir=str(directory), runtime_artifact="",
                              fullres_mask="", lowres_logits="", target="", **kwargs)


def source_manifest(root, directory, selection, *, mapped=False):
    source_dir = root / "sources"
    source_dir.mkdir()
    (source_dir / "I0000.BIN").write_bytes((directory / "INPUTQ.BIN").read_bytes())
    if mapped:
        raw = np.fromfile(directory / "TARGET.BIN", dtype=np.uint8).reshape(selection.mask_shape)
    else:
        y, x = np.indices(selection.source_shape[:2])
        raw = ((y + 3 * x) % 19).astype(np.uint8)
        raw[0, 0] = 255
    raw.tofile(source_dir / "T0000.BIN")
    manifest = json.loads((directory / "MODEL.JSN").read_text(encoding="utf-8"))
    sources = {
        "format": "P7_PREQUANTIZED_VAL_SOURCES_V1",
        "runtime_model_sha256": evaluator.sha256_file(directory / "MODEL.JSN"),
        "runtime": selection.to_manifest(),
        "preprocessing": manifest["preprocessing"],
        "samples": [{"name": "tiny-validation-sample",
                     "input": identity(source_dir / "I0000.BIN"),
                     "target": identity(source_dir / "T0000.BIN"),
                     "target_shape_hw": list(raw.shape),
                     "target_encoding": "mapped_output" if mapped else "cityscapes_train_ids",
                     "center_square_source": False}],
    }
    path = source_dir / "sources.json"
    path.write_text(json.dumps(sources), encoding="utf-8")
    return path, sources, raw


def export_args(directory, sources, output):
    return argparse.Namespace(runtime_artifact=str(directory), source_manifest=str(sources),
                              out_dir=str(output), first_index=0, limit=1, copy_param=True,
                              clean=False, clean_board_outputs=False, check_single=False,
                              write_npy=False, progress_every=0)


class UnifiedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(evaluator, "load_runtime_contract"),
                        "missing MODEL.JSN-bound unified evaluation entry point")

    def test_native_and_lite_masks_use_declared_output_grid_without_legacy_cache(self):
        for profile, classes, src, mode in ((0, 2, 0, 0), (4, 2, 0, 0), (3, 20, 0, 0),
                                            (0, 2, 1, 0), (0, 2, 1, 1), (0, 2, 2, 1),
                                            (0, 2, 3, 1)):
            with self.subTest(profile=profile, src=src, mode=mode), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "model"
                selection = geometry.runtime_selection(profile, classes, src_mode=src, mask_mode=mode)
                artifact(directory, selection)
                args = eval_args(directory)
                args.fullres_mask = str(directory / "MASK.BIN")
                args.lowres_logits = str(directory / "LOGITS.BIN")
                result = evaluator.evaluate(args)
                self.assertEqual(result["evaluation"], "unified")
                self.assertEqual(result["runtime"], selection.to_manifest())
                self.assertEqual(result["hls_lowres_logits"]["shape_hwc"],
                                 list(selection.profile.geometry.lowres_shape(classes)))
                self.assertEqual(result["hls_fullres_mask"]["mask_diff_vs_golden"]["mismatches"], 0)
                self.assertEqual(result["hls_lowres_logits"]["logit_diff"]["byte_mismatches"], 0)
                self.assertEqual(result["golden_fullres_metrics"]["pixel_accuracy"], 1.0)
                self.assertEqual(result["golden_fullres_metrics"]["valid_pixels"], selection.mask_bytes - 1)
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    evaluator.print_summary(result)
                self.assertIn("ROI", output.getvalue())

    def test_explicit_runtime_path_never_falls_back_to_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"deployment": {"class_count": 2}}))
            args = eval_args(root)
            args.runtime_artifact = str(root)
            with self.assertRaisesRegex((ValueError, FileNotFoundError), "MODEL.JSN"):
                evaluator.evaluate(args)

    def test_rejects_tampered_files_and_runtime_and_deployment_disagreements(self):
        changes = ("hash", "crc", "roi", "deployment", "semantics", "missing_input", "target_size")
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "model"
                selection = geometry.runtime_selection(0, 2, src_mode=1, mask_mode=1)
                manifest = artifact(directory, selection)
                if change == "hash":
                    (directory / "INPUTQ.BIN").write_bytes(bytes(selection.input_bytes))
                elif change == "crc":
                    manifest["files"]["logits"]["crc32"] = "00000000"
                elif change == "roi":
                    manifest["runtime"]["roi_yxhw"][1] = 0
                elif change == "deployment":
                    manifest["deployment"]["class_count"] = 20
                elif change == "semantics":
                    manifest["files"]["logits"]["semantics"] = "logits_CHW_int8"
                elif change == "missing_input":
                    del manifest["files"]["input"]
                else:
                    (directory / "TARGET.BIN").write_bytes(b"\0")
                    manifest["files"]["target"] = identity(directory / "TARGET.BIN", "mapped_target_HW_uint8")
                write_manifest(directory, manifest)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    evaluator.evaluate(eval_args(directory))

    def test_wrong_board_lengths_invalid_ids_and_wrong_roi_view_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "model"
            selection = geometry.runtime_selection(0, 2, src_mode=1, mask_mode=1)
            artifact(directory, selection)
            args = eval_args(directory)
            board = root / "BOARD.BIN"
            args.fullres_mask = str(board)
            for data in (bytes(240 * 240), bytes(256 * 512), bytes([255]) * selection.mask_bytes):
                board.write_bytes(data)
                with self.assertRaises(ValueError):
                    evaluator.evaluate(args)
            args.fullres_mask = ""
            args.lowres_logits = str(board)
            board.write_bytes(bytes(30 * 30 * 20))
            with self.assertRaises(ValueError):
                evaluator.evaluate(args)

    def test_mask_diff_includes_ignored_pixels_and_logits_are_hwc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "model"
            artifact(directory, geometry.runtime_selection(3, 20))
            board = np.fromfile(directory / "MASK.BIN", dtype=np.uint8)
            board[0] = (int(board[0]) + 1) % 20
            board.tofile(root / "BOARD.BIN")
            logits = np.fromfile(directory / "LOGITS.BIN", dtype=np.int8).reshape(32, 64, 20)
            logits[0, 0, 7] += 1
            logits.tofile(root / "LOG.BIN")
            args = eval_args(directory)
            args.fullres_mask, args.lowres_logits = str(root / "BOARD.BIN"), str(root / "LOG.BIN")
            result = evaluator.evaluate(args)
            self.assertEqual(result["hls_fullres_mask"]["mask_diff_vs_golden"]["mismatches"], 1)
            self.assertEqual(result["hls_fullres_mask"]["metrics"]["pixel_accuracy"], 1.0)
            self.assertEqual(result["hls_lowres_logits"]["logit_diff"]["byte_mismatches"], 1)
            self.assertEqual(len(result["golden_fullres_metrics"]["per_class_IoU"]), 19)

    def test_registered_original_gt_uses_half_pixel_roi_and_explicit_raw_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "model"
            selection = geometry.runtime_selection(0, 2, src_mode=1, mask_mode=0)
            manifest = artifact(directory, selection, target=False)
            # Only IDs 0/1: these are RAW train IDs, not already-mapped binary labels.
            y, x = np.indices((512, 1024))
            raw = ((x + y) % 2).astype(np.uint8)
            np.save(root / "raw.npy", raw)
            manifest["target"] = dict(available=False, original_path=str(root / "raw.npy"),
                                      original_sha256=evaluator.sha256_file(root / "raw.npy"),
                                      sampling="half_pixel_nearest_source_then_center_ROI_then_mask",
                                      center_square_source=False)
            write_manifest(directory, manifest)
            result = evaluator.evaluate(eval_args(directory))
            confusion = result["golden_fullres_metrics"]["confusion"]
            self.assertEqual(confusion[0]["support"], 0)
            self.assertEqual(confusion[1]["support"], selection.mask_bytes)
            self.assertEqual(result["ground_truth"]["mask_shape_hw"], [240, 240])

    def test_missing_or_unregistered_gt_is_not_inferred_from_target_npy_or_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "model"
            manifest = artifact(directory, geometry.runtime_selection(0, 2), target=False)
            np.save(directory / "target.npy", np.zeros((1, 240, 240), dtype=np.uint8))
            for info in ({"available": False}, {"original_path": "unknown.png", "sampling": "INTER_NEAREST"}):
                manifest["target"] = info
                write_manifest(directory, manifest)
                with self.assertRaisesRegex(ValueError, "TARGET.BIN|registered"):
                    evaluator.evaluate(eval_args(directory))

    def test_invalid_mapped_target_and_rehashed_inconsistent_golden_are_rejected(self):
        for filename, role, semantics in (("TARGET.BIN", "target", "mapped_target_HW_uint8"),
                                           ("MASK.BIN", "mask", "mask_HW_uint8_logits_then_argmax"),
                                           ("ENCODER.BIN", "encoder", "encoder_HWC_int8_selected")):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "model"
                manifest = artifact(directory, geometry.runtime_selection(0, 2))
                data = bytearray((directory / filename).read_bytes())
                data[5] = 7 if role == "target" else data[5] ^ 1
                (directory / filename).write_bytes(data)
                manifest["files"][role] = identity(directory / filename, semantics)
                write_manifest(directory, manifest)
                with self.assertRaises(ValueError):
                    evaluator.evaluate(eval_args(directory))

    def test_malformed_metadata_reports_a_contract_error(self):
        for key, value in (("deployment", None), ("preprocessing", []), ("target", []),
                           ("provenance", {})):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "model"
                manifest = artifact(directory, geometry.runtime_selection(0, 2))
                manifest[key] = value
                write_manifest(directory, manifest)
                with self.assertRaisesRegex(ValueError, "MODEL.JSN|metadata"):
                    evaluator.evaluate(eval_args(directory))

    def test_original_gt_uses_independent_exact_half_pixel_indices(self):
        for center in (False, True):
            with self.subTest(center=center), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                directory = root / "model"
                selection = geometry.runtime_selection(0, 2, src_mode=1, mask_mode=0)
                manifest = artifact(directory, selection, target=False)
                y, x = np.indices((513, 1025))
                raw = ((7 * x + 3 * y) % 19).astype(np.uint8)
                raw[::17, ::23] = 255
                np.save(root / "raw.npy", raw)
                manifest["target"] = dict(original_path=str(root / "raw.npy"),
                                          original_sha256=evaluator.sha256_file(root / "raw.npy"),
                                          sampling="half_pixel_nearest_source_then_center_ROI_then_mask",
                                          center_square_source=center)
                write_manifest(directory, manifest)
                _, _, contract = evaluator.load_runtime_contract(directory)
                target, _ = evaluator.load_runtime_target(directory, manifest, selection, contract)
                plane = raw[:, 256:769] if center else raw
                sh, sw, _ = selection.source_shape
                yy = ((2 * np.arange(sh) + 1) * plane.shape[0]) // (2 * sh)
                xx = ((2 * np.arange(sw) + 1) * plane.shape[1]) // (2 * sw)
                plane = plane[yy[:, None], xx[None, :]][:, 128:384]
                axis = ((2 * np.arange(240) + 1) * 256) // 480
                plane = plane[axis[:, None], axis[None, :]]
                expected = np.where(plane == 255, 255, np.where((plane >= 13) & (plane <= 18), 0, 1))
                np.testing.assert_array_equal(target, expected)

    def test_cli_default_metrics_path_follows_runtime_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "model"
            artifact(directory, geometry.runtime_selection(0, 2))
            with mock.patch("sys.argv", ["eval_single_hw_outputs.py", "--runtime-artifact", str(directory / "MODEL.JSN")]), \
                    contextlib.redirect_stdout(io.StringIO()), mock.patch.object(Path, "mkdir"), \
                    mock.patch.object(Path, "write_text", autospec=True) as write:
                evaluator.main()
            self.assertEqual(write.call_count, 1)
            self.assertEqual(write.call_args.args[0], directory / "single_hw_output_metrics.json")
            self.assertIn('"evaluation": "unified"', write.call_args.args[1])


class UnifiedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(dataset, "export_runtime_val_dataset"),
                        "missing explicit --runtime-artifact dataset export")

    def test_exports_source_encoder_logits_mask_and_exact_output_gt(self):
        for profile, classes, src, mode in ((0, 2, 0, 0), (0, 2, 1, 0), (0, 2, 1, 1),
                                            (0, 2, 2, 1), (3, 20, 0, 0)):
            with self.subTest(profile=profile, src=src, mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                directory = root / "model"
                selection = geometry.runtime_selection(profile, classes, src_mode=src, mask_mode=mode)
                artifact(directory, selection)
                sources, _, raw = source_manifest(root, directory, selection)
                output = root / "output"
                logits = np.fromfile(directory / "LOGITS.BIN", dtype=np.int8).reshape(
                    selection.profile.geometry.lowres_shape(classes))

                def replay(_blob, input_path, stop):
                    self.assertEqual(stop, 74)
                    self.assertEqual(Path(input_path).read_bytes(), (directory / "ENCODER.BIN").read_bytes())
                    return argparse.Namespace(tensors={17: logits})

                with mock.patch("hw_param_replay.replay_prefix", side_effect=replay), \
                        mock.patch.object(dataset, "load_exported_fake_quant_model", side_effect=AssertionError("no model/QAT")):
                    result = dataset.export_val_dataset(export_args(directory, sources, output))
                self.assertEqual(result["runtime"], selection.to_manifest())
                self.assertEqual(result["shapes"]["input_nhwc"], [1, *selection.source_shape])
                self.assertEqual(result["shapes"]["mask_hw"], list(selection.mask_shape))
                self.assertEqual(result["num_exported"], 1)
                prefix = dataset.profile_contract("binary2" if classes == 2 else "cityscapes20")
                self.assertEqual((output / f"{prefix['input_prefix']}0000.BIN").read_bytes(),
                                 (directory / "INPUTQ.BIN").read_bytes())
                expected = geometry.runtime_target_view(raw, selection, center_square_source=False)
                if classes == 2:
                    expected = np.where(expected == 255, 255, np.where((expected >= 13) & (expected <= 18), 0, 1))
                else:
                    expected = np.where(expected == 255, 19, expected)
                actual = np.fromfile(output / f"{prefix['target_prefix']}0000.BIN", dtype=np.uint8)
                np.testing.assert_array_equal(actual.reshape(selection.mask_shape), expected)
                self.assertEqual((output / f"{prefix['mask_prefix']}0000.BIN").read_bytes(),
                                 (directory / "MASK.BIN").read_bytes())
                self.assertEqual((output / "PARAM.BIN").read_bytes(), (directory / "PARAM.BIN").read_bytes())
                self.assertTrue((output / "FILEHASH.CSV").is_file())

    def test_one_frozen_native_sample_replays_without_training_or_quantization_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "model"
            selection = geometry.runtime_selection(0, 2)
            artifact(directory, selection, frozen_logits=True)
            sources, _, _ = source_manifest(root, directory, selection, mapped=True)
            args = export_args(directory, sources, root / "output")
            args.check_single = True
            with mock.patch.object(dataset, "load_exported_fake_quant_model", side_effect=AssertionError("no model/QAT")):
                result = dataset.export_val_dataset(args)
            self.assertEqual(result["reference_metrics"]["pixel_accuracy"], 1.0)
            self.assertTrue(all(item["mismatches"] == 0 for item in result["single_sample_check"]))
            self.assertEqual((root / "output/R0000.BIN").read_bytes(), (directory / "LOGITS.BIN").read_bytes())

    def test_unbound_source_quantization_preprocessing_view_and_identity_are_rejected(self):
        for change in ("model", "runtime", "preprocessing", "input_hash", "target_hash", "no_transform", "raw_labels"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                directory = root / "model"
                selection = geometry.runtime_selection(0, 2, src_mode=1, mask_mode=1)
                artifact(directory, selection)
                path, sources, _ = source_manifest(root, directory, selection)
                if change == "model":
                    sources["runtime_model_sha256"] = "0" * 64
                elif change == "runtime":
                    sources["runtime"]["mask_mode"] = 0
                elif change == "preprocessing":
                    sources["preprocessing"]["mean"] = [123]
                elif change == "input_hash":
                    sources["samples"][0]["input"]["sha256"] = "0" * 64
                elif change == "target_hash":
                    sources["samples"][0]["target"]["sha256"] = "0" * 64
                elif change == "no_transform":
                    del sources["samples"][0]["center_square_source"]
                else:
                    (path.parent / "T0000.BIN").write_bytes(bytes([42]) * (256 * 512))
                    sources["samples"][0]["target"] = identity(path.parent / "T0000.BIN")
                path.write_text(json.dumps(sources), encoding="utf-8")
                output = root / "output"
                with mock.patch("hw_param_replay.replay_prefix", side_effect=AssertionError("reject before replay")):
                    with self.assertRaises(ValueError):
                        dataset.export_val_dataset(export_args(directory, path, output))
                self.assertFalse(output.exists(), "preflight failure must not leave successful-looking output")

    def test_raw_generation_is_rejected_without_explicit_bound_source_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "model"
            artifact(directory, geometry.runtime_selection(0, 2, src_mode=2, mask_mode=1))
            args = export_args(directory, "", root / "output")
            with self.assertRaisesRegex(ValueError, "source-manifest.*prequantized|prequantized.*source-manifest"):
                dataset.export_val_dataset(args)
            self.assertFalse((root / "output").exists())

    def test_runtime_cli_contract_is_explicit_and_has_no_legacy_default_overrides(self):
        with mock.patch("sys.argv", ["export_val_hw_dataset.py", "--runtime-artifact", "M", "--source-manifest", "S"]):
            args = dataset.parse_args()
        self.assertEqual(args.runtime_artifact, str(Path("M").resolve()))
        self.assertEqual(args.source_manifest, str(Path("S").resolve()))

    def test_unified_artifact_cannot_enter_legacy_export_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "model"
            artifact(directory, geometry.runtime_selection(0, 2))
            args = argparse.Namespace(artifact_dir=str(directory))
            with self.assertRaisesRegex(ValueError, "--runtime-artifact"):
                dataset.export_val_dataset(args)

    def test_replay_chw_or_unsigned_outputs_never_publish_a_manifest(self):
        for shape, dtype in (((2, 30, 30), np.int8), ((30, 30, 2), np.uint8)):
            with self.subTest(shape=shape, dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                directory = root / "model"
                selection = geometry.runtime_selection(0, 2)
                artifact(directory, selection)
                sources, _, _ = source_manifest(root, directory, selection)
                with mock.patch("hw_param_replay.replay_prefix", return_value=argparse.Namespace(
                        tensors={17: np.zeros(shape, dtype=dtype)})):
                    with self.assertRaisesRegex(ValueError, "HWC signed INT8"):
                        dataset.export_val_dataset(export_args(directory, sources, root / "output"))
                self.assertFalse((root / "output/manifest.json").exists())
                self.assertFalse((root / "output/SDMANIFEST.JSON").exists())

    def test_rgb_lite_structural_sources_preserve_channels_and_gt_roi(self):
        # Structural transforms only: no claim that an adapted RGB240 model exists.
        for source_mode in (4, 5, 6):
            for mask_mode in (0, 1):
                with self.subTest(source=source_mode, mask=mask_mode), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    selection = geometry.runtime_selection(1, 20, src_mode=source_mode, mask_mode=mask_mode)
                    source = np.arange(selection.input_bytes, dtype=np.int64).astype(np.int8).reshape(selection.source_shape)
                    source.tofile(root / "SOURCE.BIN")
                    y, x = np.indices(selection.source_shape[:2])
                    raw = ((y + x) % 20).astype(np.uint8)
                    raw[0, 0] = 255
                    raw.tofile(root / "GT.BIN")
                    sample = dict(name="structural", input=identity(root / "SOURCE.BIN"),
                                  target=identity(root / "GT.BIN"), target_shape_hw=list(raw.shape),
                                  target_encoding="cityscapes_train_ids", center_square_source=False)
                    contract = evaluator.DeploymentContract("cityscapes20", 20, 19, selection.profile.geometry)
                    actual_source, actual_gt = dataset.load_bound_source_sample(root, sample, selection, contract)
                    np.testing.assert_array_equal(actual_source, source)
                    encoder = geometry.select_runtime_input(actual_source, selection)
                    top, left, h, w = selection.roi
                    axis = ((2 * np.arange(240) + 1) * h) // 480
                    np.testing.assert_array_equal(encoder, source[top + axis[:, None], left + axis[None, :], :])
                    mh, mw = selection.mask_shape
                    yy = top + ((2 * np.arange(mh) + 1) * h) // (2 * mh)
                    xx = left + ((2 * np.arange(mw) + 1) * w) // (2 * mw)
                    expected = raw[yy[:, None], xx[None, :]]
                    np.testing.assert_array_equal(actual_gt, np.where(expected == 255, 19, expected))


@unittest.skipUnless((ROOT / "hw_artifacts/unified_int8/MODELS.JSN").is_file(),
                     "published unified catalog not present")
class BoardValidationSuiteTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(dataset, "publish_board_validation_suite"),
                        "missing VAL500 suite publisher for the unified App")

    def fixture(self, root):
        model_dir = root / "model"
        selection = geometry.runtime_selection(0, 2)
        model = artifact(model_dir, selection)
        data_dir = root / "dataset"
        data_dir.mkdir()
        rows = []
        for i in range(500):
            row = dict(index=i, image=f"val-image-{i:04d}", input=f"B{i:04d}.BIN", target=f"T{i:04d}.BIN")
            for role, frozen in (("input", "INPUTQ.BIN"), ("target", "TARGET.BIN")):
                os.link(model_dir / frozen, data_dir / row[role])
                record = identity(model_dir / frozen)
                row[f"{role}_bytes"] = record["bytes"]
                row[f"{role}_sha256"] = record["sha256"]
            rows.append(row)
        manifest = dict(format="P7_UNIFIED_INT8_VAL_DATASET_V1", runtime=selection.to_manifest(),
                        runtime_artifact=str(model_dir), runtime_model_sha256=evaluator.sha256_file(model_dir/"MODEL.JSN"),
                        param_sha256=model["files"]["param"]["sha256"], deployment=model["deployment"],
                        preprocessing=model["preprocessing"], num_exported=500, num_val_total=500, samples=rows)
        path = data_dir / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        spec = root / "suite.json"
        spec.write_text(json.dumps(dict(format="P7_UNIFIED_VAL500_SUITE_V1",
                                       datasets=[dict(key="G240B2", dataset=str(path))])), encoding="utf-8")
        return spec, path, manifest

    def test_packages_exact_500_short_names_and_binds_app_crcs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec, _, _ = self.fixture(root)
            result = dataset.publish_board_validation_suite(spec, root/"board")
            self.assertEqual(result["case_count"], 1)
            self.assertEqual(result["samples_per_case"], 500)
            rows = list(csv.DictReader(io.StringIO((root/"board/VALRUNS.CSV").read_text())))
            self.assertEqual(rows[0]["key"], "G240B2")
            index = root/"board/G240B2/SAMPLES.CSV"
            self.assertEqual(rows[0]["samples_crc"], f"{zlib.crc32(index.read_bytes()):08X}")
            self.assertEqual(len(list(csv.DictReader(io.StringIO(index.read_text())))), 500)
            self.assertTrue((root/"board/G240B2/I0499.BIN").is_file())
            self.assertTrue((root/"board/G240B2/T0499.BIN").is_file())
            self.assertFalse((root/"board/G240B2/O0000.BIN").exists())
            self.assertEqual(len(result["not_in_suite"]), 39)

    def test_rejects_incomplete_repeated_reordered_or_unbound_validation_before_publish(self):
        for mutation in ("incomplete", "duplicate", "index", "quantization", "geometry", "traversal", "hash"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                spec, path, data = self.fixture(root)
                if mutation == "incomplete": data["samples"].pop()
                elif mutation == "duplicate": data["samples"][1]["image"] = data["samples"][0]["image"]
                elif mutation == "index": data["samples"][1]["index"] = 2
                elif mutation == "quantization": data["runtime_model_sha256"] = "0"*64
                elif mutation == "geometry": data["runtime"]["mask_shape_hw"] = [512, 512]
                elif mutation == "traversal": data["samples"][0]["input"] = "../model/INPUTQ.BIN"
                else: data["samples"][0]["target_sha256"] = "0"*64
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    dataset.publish_board_validation_suite(spec, root/"board")
                self.assertFalse((root/"board").exists())


class PublishedUnifiedIntegrationTests(unittest.TestCase):
    def test_every_registered_case_validates_and_only_registered_gt_enables_metrics(self):
        catalog_dir = ROOT / "hw_artifacts/unified_int8"
        catalog = json.loads((catalog_dir / "MODELS.JSN").read_text(encoding="utf-8"))
        cases = [case for model in catalog["models"] for case in model["cases"]]
        self.assertTrue(cases)
        for name in cases:
            with self.subTest(case=name):
                directory = catalog_dir / name
                args = eval_args(directory)
                args.fullres_mask = str(directory / "MASK.BIN")
                args.lowres_logits = str(directory / "LOGITS.BIN")
                model, _, _ = evaluator.load_runtime_contract(directory)
                if "target" not in model["files"]:
                    self.assertFalse(model["target"]["available"])
                    with self.assertRaisesRegex(ValueError, "TARGET.BIN|registered"):
                        evaluator.evaluate(args)
                    continue
                result = evaluator.evaluate(args)
                self.assertEqual(result["hls_fullres_mask"]["mask_diff_vs_golden"]["mismatches"], 0)
                self.assertEqual(result["hls_lowres_logits"]["logit_diff"]["byte_mismatches"], 0)
                self.assertEqual(result["ground_truth"]["path"], str(directory / "TARGET.BIN"))
                self.assertGreater(result["golden_fullres_metrics"]["valid_pixels"], 0)

    def test_one_published_lite_source_exports_identical_integer_references(self):
        directory = ROOT / "hw_artifacts/unified_int8/G2S2M1"
        model = json.loads((directory / "MODEL.JSN").read_text(encoding="utf-8"))
        selection = geometry.runtime_selection_from_manifest(model)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources, _, _ = source_manifest(root, directory, selection, mapped=True)
            args = export_args(directory, sources, root / "output")
            args.check_single = True
            with mock.patch.object(dataset, "load_exported_fake_quant_model", side_effect=AssertionError("no model/QAT")):
                result = dataset.export_val_dataset(args)
            self.assertEqual(result["num_exported"], 1)
            self.assertTrue(all(check["mismatches"] == 0 for check in result["single_sample_check"]))
            self.assertEqual((root / "output/T0000.BIN").read_bytes(), (directory / "TARGET.BIN").read_bytes())


if __name__ == "__main__":
    unittest.main()
