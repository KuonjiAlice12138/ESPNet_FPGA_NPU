"""Deployment geometry checks for the grayscale H240 branch."""

import inspect
import json
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.geometry_contract import DeploymentGeometry
from tools import export_int8_hw_blob as compiler
from tools import eval_single_hw_outputs as single_eval
from tools import export_val_hw_dataset as val_export


class GrayGeometryContractTests(unittest.TestCase):
    def test_gray_geometry_serializes_channel_count(self):
        self.assertIn("input_channels", inspect.signature(DeploymentGeometry).parameters)
        geometry = DeploymentGeometry(240, 240, 8, input_channels=1)
        self.assertEqual(geometry.input_shape_nhwc, (240, 240, 1))
        self.assertEqual(geometry.input_bytes, 57600)
        self.assertEqual(DeploymentGeometry.from_manifest(geometry.to_manifest()), geometry)

    def test_legacy_manifests_default_to_rgb(self):
        geometry = DeploymentGeometry.from_manifest({
            "input_height": 256, "input_width": 512, "target_scale": 8,
            "logits_height": 32, "logits_width": 64,
        })
        self.assertIn("input_channels", inspect.signature(DeploymentGeometry).parameters)
        self.assertEqual(geometry.input_channels, 3)
        self.assertEqual(geometry.input_bytes, 393216)

    def test_non_image_channel_counts_are_rejected(self):
        self.assertIn("input_channels", inspect.signature(DeploymentGeometry).parameters)
        for count in (0, 2, 4):
            with self.assertRaises(ValueError):
                DeploymentGeometry(240, 240, input_channels=count)


class GrayCompilerContractTests(unittest.TestCase):
    geometry = DeploymentGeometry(240, 240, 8, input_channels=1)

    def tearDown(self):
        compiler.activate_deployment_plans(geometry=compiler.DEFAULT_GEOMETRY)

    def test_tensor_plan_uses_actual_grayscale_channels(self):
        tensors = compiler.build_tensor_plan(self.geometry)
        self.assertEqual([tensors[i].c for i in (0, 1, 8, 18)], [1, 1, 1, 1])
        self.assertEqual([tensors[i].c for i in (2, 3, 9, 10)], [17, 17, 129, 129])

    def test_gray_memory_layout_reserves_pool_tail_and_aligned_scratch(self):
        layout = compiler.build_memory_layout(self.geometry)
        self.assertEqual(layout.fmbuf_bytes, 848000)
        self.assertEqual(layout.uram_bytes, 819200)
        self.assertEqual(layout.residual_base, 491520)
        self.assertEqual(layout.c1_scratch_base, 753664)
        self.assertEqual(layout.pool1_bytes, 14400)
        self.assertEqual(layout.pool2_bytes, 3616)
        self.assertEqual(layout.l3_scratch_slot_bytes, 23040)

    def test_activation_plans_include_gray_affine_and_uop_channels(self):
        compiler.activate_deployment_plans(geometry=self.geometry)
        self.assertEqual([compiler.AFFINES[i].channels for i in (0, 3)], [17, 129])
        uops = compiler.build_uops(geometry=self.geometry)
        convs = {u.param_id: u for u in uops if u.opcode == compiler.UOP_CONV}
        self.assertEqual([convs[i].in_c for i in (0, 1, 13)], [1, 17, 129])

    def test_gray_modes_have_new_v4_ids(self):
        for channels, expected in ((1, 12), (17, 13), (129, 14)):
            name = f"WIN_MODE_3X3_STAGED_C{channels}"
            self.assertEqual(getattr(compiler, name, None), expected)
            self.assertEqual(compiler.window_mode_for(channels, 3), expected)

    def test_compact_rows_have_explicit_padding_contract(self):
        tensors = compiler.build_tensor_plan(self.geometry)
        self.assertTrue(getattr(tensors[2], "row_align32", False))
        self.assertEqual(tensors[2].row_stride_bytes, 2048)
        self.assertEqual(tensors[9].row_stride_bytes, 7744)
        self.assertEqual(tensors[0].row_stride_bytes, 240)
        self.assertEqual(tensors[1].row_stride_bytes, 120)
        self.assertEqual(tensors[2].packed_phys_c, 0x8011)

    def test_first_conv_fuses_concat_before_selecting_store_layout(self):
        compiler.activate_deployment_plans(geometry=self.geometry)
        uops = compiler.build_uops(geometry=self.geometry)
        _, _, schedule_ids, _ = compiler.build_window_schedule_sections(uops)
        convs = [u for u in uops if u.opcode == compiler.UOP_CONV]
        offsets = {u.param_id: u.param_id * 1024 for u in convs}
        weights = [{
            "param_id": u.param_id,
            "k_tiles": (u.in_c * u.kernel * u.kernel + 31) // 32,
            "packed_words": ((u.in_c * u.kernel * u.kernel + 31) // 32) * u.out_c,
        } for u in convs]
        descs, consumers, _, entries, _, _ = compiler.build_exec_plan_sections(
            uops, schedule_ids, offsets, weights
        )
        self.assertEqual(len(entries), 16)
        first = next(d for d in descs if d.param_id == 0)
        consumer = consumers[first.row_consumer_id]
        self.assertEqual(consumer.mode, compiler.ROW_CONSUMER_CAT_AFFINE_STORE)
        self.assertEqual(consumer.valid_c, 17)
        self.assertEqual(consumer.reserved0, compiler.STORE_LAYOUT_COMPACT_C17)

    def test_block5_scratch_audit_includes_row_padding(self):
        audit = compiler.build_block5_feasibility_audit(self.geometry)
        self.assertEqual(audit["stages"][0]["first4_compact_bytes"], 190080)
        self.assertEqual(audit["stages"][1]["first4_compact_bytes"], 95040)


class GrayEvaluationContractTests(unittest.TestCase):
    def test_binary_target_mapping_does_not_remap_already_mapped_labels(self):
        target = torch.tensor([0, 1, 255])
        self.assertTrue(torch.equal(val_export.map_target_for_profile(
            target, "binary2", already_mapped=True), target))
        self.assertTrue(torch.equal(val_export.map_target_for_profile(target, "binary2"),
                                    torch.tensor([1, 1, 255])))

    def test_fullres_target_uses_declared_center_crop(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.ones((100, 200), dtype=np.uint8)
            labels[:, 50:150] = 13
            cv2.imwrite(str(root / "target.png"), labels)
            (root / "cache.pkl").write_bytes(pickle.dumps({"valAnnot": [str(root / "target.png")]}))
            (root / "manifest.json").write_text(json.dumps({
                "deployment": {"profile_name": "binary2", "class_count": 2},
                "geometry": DeploymentGeometry(240, 240, 8, 1).to_manifest(),
                "preprocessing": {"image_mode": "gray", "center_square_crop": True},
            }), encoding="utf-8")
            contract = single_eval.load_deployment_contract(root)
            target = single_eval.load_fullres_target(root / "cache.pkl", 0, contract, root)
            self.assertEqual(target.shape, (240, 240))
            self.assertTrue(np.all(target == 0))

    def test_fullres_exported_target_can_be_reduced_to_logits_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.npy"
            target = np.indices((240, 240))[1].astype(np.uint8) % 2
            np.save(path, target[np.newaxis])
            contract = single_eval.DeploymentContract(
                "binary2", 2, None, DeploymentGeometry(240, 240, 8, 1)
            )
            low = single_eval.load_lowres_target(path, contract)
            self.assertEqual(low.shape, (30, 30))
            np.testing.assert_array_equal(low, target[::8, ::8])

    def test_smoke_input_name_comes_from_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke = root / "val_smoke"
            smoke.mkdir()
            (root / "input_q.bin").write_bytes(b"gray input")
            (smoke / "B0000.BIN").write_bytes(b"gray input")
            (smoke / "manifest.json").write_text(json.dumps({
                "files": {"input_pattern": "B%04d.BIN"},
            }), encoding="utf-8")
            contract = single_eval.DeploymentContract(
                "binary2", 2, None, DeploymentGeometry(240, 240, 8, 1)
            )
            checks = single_eval.check_single_vs_val_smoke(root, smoke, contract)
            self.assertTrue(checks["input_vs_I0000"]["same_sha256"])
            self.assertEqual(Path(checks["input_vs_I0000"]["right"]).name, "B0000.BIN")


class GrayHlsContractTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "ESP_INT8_hls"

    def test_fixed_frame_contract_is_gray_h240(self):
        config = (self.root / "include/npu_config.hpp").read_text(encoding="utf-8-sig")
        for name, value in (("INPUT_FRAME_H", 240), ("INPUT_FRAME_W", 240),
                            ("INPUT_FRAME_C", 1), ("ENCODER_OUT_H", 30),
                            ("ENCODER_OUT_W", 30)):
            self.assertIn(f"constexpr int {name} = {value};", config)
        top = (self.root / "src/int8_core.cpp").read_text(encoding="utf-8-sig")
        self.assertIn("bundle=gmem0 depth=1800", top)
        self.assertIn("bundle=gmem1 depth=1800", top)

    def test_tensor_stride_decodes_compiler_row_padding_flag(self):
        types = (self.root / "include/npu_types.hpp").read_text(encoding="utf-8-sig")
        self.assertIn("tensor_row_stride_bytes(", types)
        self.assertIn("0x8000", types)
        self.assertIn("0x7fff", types)
        for name in ("memory.cpp", "win_gen.cpp", "ppu.cpp", "vec_alu_engine.cpp", "conv_store.cpp"):
            source = (self.root / "src" / name).read_text(encoding="utf-8-sig")
            self.assertNotIn("? desc.c : desc.reserved0", source)

    def test_old_rgb_kernels_do_not_remain_in_gray_owners(self):
        win = (self.root / "src/win_gen.cpp").read_text(encoding="utf-8-sig")
        for channels in (1, 17, 129):
            self.assertIn(f"build_staged_c{channels}_word(", win)
        for token in ("build_staged_c3_word(", "build_staged_c19_word(",
                      "build_staged_c131_word(", "s_narrow_reuse_", "c3_reuse_incremental"):
            self.assertNotIn(token, win)
        pool = (self.root / "src/avgpool_unit.cpp").read_text(encoding="utf-8-sig")
        self.assertNotIn("avgpool_unit_c3_fast", pool)
        self.assertIn("avgpool_unit_c1", pool)
        up = (self.root / "src/upsample_unit.cpp").read_text(encoding="utf-8-sig")
        self.assertNotIn("FULLRES_MASK_W / AXI_WORD_BYTES", up)

    def test_retired_dump_hooks_and_buffer_constants_are_removed(self):
        for name in ("int8_core.cpp", "main_ctrl.cpp", "conv_engine.cpp", "ppu.cpp"):
            source = (self.root / "src" / name).read_text(encoding="utf-8-sig")
            for token in ("csim_dump_u40_prestore_row", "csim_dump_tensor_set_",
                          "CSIM_DUMP_UPSAMPLE_INPUT", "s_csim_tensor_dumped"):
                self.assertFalse(token in source, f"{name}: retired token {token}")
        config = (self.root / "include/npu_config.hpp").read_text(encoding="utf-8-sig")
        self.assertNotIn("QBUF", config)
        self.assertNotIn("SCRATCH_BRAM_BYTES", config)


if __name__ == "__main__":
    unittest.main()
