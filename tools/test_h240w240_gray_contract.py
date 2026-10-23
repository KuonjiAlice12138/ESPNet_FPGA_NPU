"""Deployment geometry checks for the grayscale H240 branch."""

import inspect
from dataclasses import replace
import json
import pickle
from pathlib import Path
import re
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
from tools import hw_param_replay as replay
from tools import check_p6_hls_structure as structure
from tools import audit_hls_reports as report_audit


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

    def test_round6_c1_compiles_eight_word_keep1(self):
        desc, _ = compiler.make_window_pack_schedule(
            1, 3, source_w=240, stride=2, dilation=1, padding=1,
            out_w=120, flags=compiler.WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)
        self.assertEqual(desc.reserved[9], 2 | (8 << 2))

    def test_round6_transfer_protocol_is_explicit_and_audited(self):
        for channels, paired in ((12, True), (17, True), (25, False)):
            for dilation in (1, 2, 4, 8, 16):
                if channels == 17 and dilation != 1:
                    continue
                desc, _ = compiler.make_window_pack_schedule(
                    channels, 3, source_w=60, stride=2 if channels == 17 else 1,
                    dilation=dilation, padding=dilation, out_w=30 if channels == 17 else 60,
                    flags=compiler.WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2 if paired else 0)
                protocol = (desc.reserved[9] >> 9) & 3
                self.assertEqual(protocol, 1 if channels == 12 and dilation == 1 else 2)
                audit = compiler.audit_window_transfer_contract(desc)
                self.assertEqual(audit["produced_words"], audit["consumed_words"])
                self.assertLess(audit["produced_words"], audit["full_window_words"])
                self.assertEqual(audit["act_words_per_stream"],
                                 ((desc.out_w + 1) // 2 if paired else desc.out_w) * desc.k_tiles)

    def test_round6_replay_rejects_bad_reuse_and_transfer_bits(self):
        path = Path(__file__).resolve().parents[1] / "hw_artifacts/binary2_int8_h240w240_gray_v4/PARAM.BIN"
        blob = replay.ParamBlob(path)
        conv = blob.conv_exec_by_index[0]
        source = blob.tensor_desc[conv.src_tensor]
        desc = blob.window_sched_by_index[conv.window_sched_id]
        words = list(desc.reserved)
        words[9] = 2 | (8 << 2)
        compiled = replace(desc, reserved=words)
        self.assertTrue(replay.window_row_reuse_contract_valid(compiled, conv, source))
        for bad in (2 | (7 << 2), 2 | (8 << 2) | (1 << 11), 1 | (8 << 2)):
            altered = list(words)
            altered[9] = bad
            self.assertFalse(replay.window_row_reuse_contract_valid(
                replace(compiled, reserved=altered), conv, source))
        self.assertFalse(replay.window_row_reuse_contract_valid(
            compiled, conv, replace(source, reserved1=1)))

    def test_round6_replay_checks_delta_and_fixed_run_contracts(self):
        blob = replay.ParamBlob(Path(__file__).resolve().parents[1] /
                                "hw_artifacts/binary2_int8_h240w240_gray_v4/PARAM.BIN")
        for desc in blob.window_sched_by_index:
            protocol = compiler.compile_window_transfer_protocol(
                desc.in_c, desc.kernel, desc.stride, desc.dilation, desc.flags)
            if not protocol:
                continue
            words = list(desc.reserved)
            words[9] = protocol << 9
            compiled = replace(desc, reserved=words)
            self.assertTrue(replay.window_row_reuse_contract_valid(compiled))
            self.assertFalse(replay.window_row_reuse_contract_valid(
                replace(compiled, flags=desc.flags | 4)))
            malformed = list(words)
            malformed[9] = 3 << 9
            self.assertFalse(replay.window_row_reuse_contract_valid(
                replace(compiled, reserved=malformed)))
            if protocol == 1:
                malformed = list(words)
                malformed[1] = 0x0300
                self.assertFalse(replay.window_row_reuse_contract_valid(
                    replace(compiled, reserved=malformed)))

    def test_round6_schedule_only_recompile_handles_scratch_sources(self):
        path = Path(__file__).resolve().parents[1] / "hw_artifacts/binary2_int8_h240w240_gray_v4/PARAM.BIN"
        source = path.read_bytes()
        blob = replay.ParamBlob(path)
        self.assertTrue(any(conv.src_tensor >= len(blob.tensor_desc) for conv in blob.conv_exec_by_index))
        compiled = compiler.compile_window_reuse_param(path)
        self.assertEqual(len(compiled), len(source))
        allowed = {blob.header["window_sched_offset"] + sid * 128 + 126 + byte
                   for sid in range(len(blob.window_sched_by_index)) for byte in (0, 1)}
        changed = {index for index, (a, b) in enumerate(zip(source, compiled)) if a != b}
        self.assertTrue(changed)
        self.assertTrue(changed <= allowed)

    def test_round6_metadata_relocates_relative_and_absolute_artifact_paths(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "hw_artifacts/binary2_int8_h240w240_gray_v4"
        target = root / "hw_artifacts/binary2_int8_h240w240_gray_r6_v4"
        metadata = {"param": str(source.relative_to(root) / "PARAM.BIN"),
                    "nested": [source.as_posix() + "/param_audit.json"]}
        result = compiler.relocate_artifact_metadata(metadata, source, target)
        self.assertEqual(Path(result["param"]).resolve(), target / "PARAM.BIN")
        self.assertEqual(Path(result["nested"][0]).resolve(), target / "param_audit.json")


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


class GrayRound7SourceTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "ESP_INT8_hls/src"

    def test_block5_has_one_four_lane_resident_word_kernel(self):
        source = (self.root / "ppu.cpp").read_text(encoding="utf-8-sig")
        self.assertIn("PPU_BLOCK_LANES = 4", source)
        self.assertIn("ppu_load_block5_affine_bank(", source)
        self.assertIn("ppu_apply_block_word(", source)
        self.assertNotIn("ppu_apply_block_add_affine_word(", source)
        self.assertNotIn("ppu_apply_block_affine_word(", source)
        self.assertNotIn("ppu_finalize_block5_emit_tiles(", source)

    def test_vec_uses_one_absolute_span_and_store_service(self):
        source = (self.root / "vec_alu_engine.cpp").read_text(encoding="utf-8-sig")
        self.assertIn("on_chip_memory_read_packed_span(", source)
        self.assertIn("tile_service_count", source)
        self.assertNotIn("read_block_affine_source_tile(", source)
        self.assertNotIn("for (int middle =", source)
        self.assertEqual(source.count("conv_store_write_row_contiguous_word("), 2)

    def test_ppu_uses_pairs_without_the_retired_group_loop(self):
        source = (self.root / "ppu.cpp").read_text(encoding="utf-8-sig")
        self.assertIn("ppu_append_compact_pair(", source)
        self.assertIn("source_cursor >>= 64", source)
        self.assertNotIn("ppu_merge_compact_group(", source)
        self.assertIn("conv_stream.read()", source)

    def test_compact_pack_memory_commit_is_outside_the_pipeline(self):
        ppu = (self.root / "ppu.cpp").read_text(encoding="utf-8-sig")
        conv = (self.root / "conv_engine.cpp").read_text(encoding="utf-8-sig")
        self.assertEqual(structure.check_round4a_ppu_structure(ppu, conv), 0)

    def test_report_audit_flags_clock_and_specialization_warnings(self):
        for message in (
            "WARNING: [HLS 200-871] Estimated clock period exceeds the target",
            "WARNING: [HLS 200-887] Cannot meet target clock period",
            "WARNING: [HLS 214-366] Duplicating function as different function signatures were detected",
        ):
            self.assertTrue(any(rule.pattern.search(message) for rule in report_audit.HIGH_RISKS),
                            message)

    def test_upsample_reuses_binary_corners_and_keeps_ties(self):
        source = (self.root / "upsample_unit.cpp").read_text(encoding="utf-8-sig")
        self.assertIn("binary_corner_difference(", source)
        self.assertIn("cached_x0", source)
        self.assertIn("difference > 0", source)
        self.assertNotIn("interpolate_argmax_label(", source)

    def test_pool_arithmetic_is_separate_from_memory_commit(self):
        source = (self.root / "avgpool_unit.cpp").read_text(encoding="utf-8-sig")
        begin = source.index("static void compute_c1_group_word(")
        end = source.index("\nstatic bool ", begin)
        arithmetic = source[begin:end]
        self.assertIn("#pragma HLS PIPELINE II=1", arithmetic)
        self.assertNotIn("on_chip_memory_", arithmetic)
        self.assertIn("round_div9(sum)", arithmetic)
        self.assertNotIn("#pragma HLS UNROLL factor=32", source)


class GrayRound7RSourceTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "ESP_INT8_hls/src"

    def test_vec_commit_is_a_short_loop_without_tile_dispatch(self):
        source = (self.root / "vec_alu_engine.cpp").read_text(encoding="utf-8-sig")
        body = structure.function_body(source, "run_fixed_affine_common")
        self.assertIn("VEC_AFFINE_TILES:", body)
        self.assertIn("VEC_COMMIT_WORDS:", body)
        commit = body.split("VEC_COMMIT_WORDS:", 1)[1].split("aligned_row +=", 1)[0]
        self.assertIn("#pragma HLS PIPELINE II=1", commit)
        self.assertIn("ROW_CONTIG_MAX_WORDS", commit)
        self.assertIn("conv_store_write_row_contiguous_word(", commit)
        for token in ("tile_service_count", "vec_select_source_span(",
                      "vec_alu_apply_affine_block(", "vec_stage_compact_word(",
                      "vec_finish_compact_row("):
            self.assertNotIn(token, commit)

    def test_vec_aligned_staging_reuses_the_single_existing_row_buffer(self):
        source = (self.root / "vec_alu_engine.cpp").read_text(encoding="utf-8-sig")
        body = structure.function_body(source, "run_fixed_affine_common")
        self.assertIn("s_shared_row_contig_words[tile.to_uint()] = out_packed;", body)
        self.assertEqual(re.findall(r"static\s+axi_vec_t\s+(\w+)\s*\[", source),
                         ["s_shared_row_contig_words"])
        self.assertEqual(source.count("vec_alu_apply_affine_block("), 2)
        self.assertEqual(source.count("on_chip_memory_read_packed_span("), 2)
        self.assertEqual(source.count("conv_store_write_row_contiguous_word("), 2)
        finish = body.index("if (row_contiguous && !vec_finish_compact_row(")
        self.assertLess(finish, body.index("VEC_COMMIT_WORDS:"))

    def test_vec_commit_uses_an_independent_bounded_incremental_cursor(self):
        source = (self.root / "vec_alu_engine.cpp").read_text(encoding="utf-8-sig")
        body = structure.function_body(source, "run_fixed_affine_common")
        self.assertIn("for (ap_uint<9> word = 0; word < ROW_CONTIG_MAX_WORDS; ++word)", body)
        self.assertIn("commit_offset += commit_step;", body)
        self.assertNotIn("VEC_MAX_ROW_SERVICES", source)
        self.assertNotIn("service_count = tile_service_count +", body)
        self.assertNotIn("service.to_uint() -", body)

    def test_vec_commit_prevalidates_the_span_without_status_feedback(self):
        source = (self.root / "vec_alu_engine.cpp").read_text(encoding="utf-8-sig")
        body = structure.function_body(source, "run_fixed_affine_common")
        self.assertTrue("conv_store_word_span_plan_ok(" in body,
                        "Vec must prove the destination span before dropping redundant per-word status")
        self.assertLess(body.index("conv_store_word_span_plan_ok("), body.index("for (int outer"))
        self.assertIn("const u64_t last_write_span", body)
        commit = body.split("VEC_COMMIT_WORDS:", 1)[1].split("aligned_row +=", 1)[0]
        self.assertIn("(void)conv_store_write_row_contiguous_word(", commit)
        self.assertNotIn("return ERR_", commit)
        self.assertNotIn("if (!conv_store_write", commit)

    def test_vec_span_validator_is_scalar_and_overflow_safe(self):
        source = (self.root / "conv_store.cpp").read_text(encoding="utf-8-sig")
        body = structure.function_body(source, "conv_store_word_span_plan_ok")
        self.assertIsNotNone(body, "The existing store owner must validate the whole word span")
        self.assertIn("const u64_t end_offset", body)
        self.assertIn("ROW_CONTIG_MAX_WORDS", body)
        self.assertIn("BRAM_SCR1_BYTES", body)
        self.assertIn("FMBUF_POOL_TMP_BASE", body)
        self.assertNotIn("on_chip_memory_", body)
        self.assertNotIn("#pragma HLS PIPELINE II=1", body)

    def test_absent_vec_source_metadata_is_not_evaluated(self):
        source = (self.root / "vec_alu_engine.cpp").read_text(encoding="utf-8-sig")
        self.assertTrue("const u32_t base2 = has_src2" in source,
                        "Two-source EXEC must not read the absent third descriptor")
        self.assertTrue("aligned_source = {0, 0, 0, 0, 0};" in source,
                        "Compact EXEC must not copy uninitialized aligned metadata")

    def test_vec_hoists_aligned_source_plan_and_keeps_one_datapath(self):
        source = (self.root / "vec_alu_engine.cpp").read_text(encoding="utf-8-sig")
        self.assertIn("vec_select_source_span(", source)
        self.assertIn("aligned_source", source)
        self.assertIn("VEC_MAX_ROW_TILES", source)
        self.assertNotIn("read_block_affine_source_span(", source)
        self.assertNotIn("MAX_FM_H * MAX_FM_W + ROW_CONTIG_MAX_WORDS", source)
        self.assertEqual(source.count("vec_alu_apply_affine_block("), 2)
        self.assertEqual(source.count("on_chip_memory_read_packed_span("), 2)
        self.assertEqual(source.count("conv_store_write_row_contiguous_word("), 2)
        finish = source.index("if (row_contiguous && !vec_finish_compact_row(", source.index("static error_code_t run_fixed_affine_common"))
        commit = source.index("VEC_COMMIT_WORDS:", finish)
        self.assertLess(finish, commit)
        self.assertNotIn("if (word_idx == 0", source)

    def test_block5_coefficient_cache_is_bound_to_param_generation_and_row(self):
        ppu = (self.root / "ppu.cpp").read_text(encoding="utf-8-sig")
        param = (self.root / "param_dma.cpp").read_text(encoding="utf-8-sig")
        self.assertIn("param_dma_generation()", ppu)
        self.assertIn("++s_param_generation;", param)
        self.assertIn("static bool affine_cache_valid = false;", ppu)
        self.assertIn("affine_cache_next_row == abs_row", ppu)
        self.assertIn("affine_cache_generation == generation", ppu)
        self.assertIn("abs_row != 0", ppu)
        self.assertIn("affine_cache_valid = false;", ppu)
        self.assertEqual(ppu.count("ppu_load_block5_affine_bank("), 2)
        self.assertIn("static i32_t qmul[PPU_BLOCK_LANES][TM];", ppu)

    def test_preadd_uses_incremental_span_without_coordinate_reader(self):
        ppu = (self.root / "ppu.cpp").read_text(encoding="utf-8-sig")
        begin = ppu.index("bool ppu_preadd_segment(")
        end = ppu.index("\nbool ppu_preadd_row(", begin)
        preadd = ppu[begin:end]
        self.assertIn("on_chip_memory_read_packed_span(", preadd)
        self.assertIn("source_offset += source_step;", preadd)
        self.assertNotIn("on_chip_memory_read_packed_tile(", preadd)
        self.assertIn("ppu_apply_add_word(", preadd)


class GrayRound8SourceTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "ESP_INT8_hls"

    def test_vec_skips_empty_pack_loops_and_has_one_staging_tail(self):
        source = (self.root / "src/vec_alu_engine.cpp").read_text(encoding="utf-8-sig")
        stage = structure.function_body(source, "vec_stage_compact_word")
        self.assertIn("if (pair_count != 0U)", stage)
        self.assertIn("if (tail_count != 0U)", stage)
        self.assertIn("VEC_PACK_PAIRS:", stage)
        self.assertIn("VEC_PACK_TAIL:", stage)
        self.assertEqual(stage.count("s_shared_row_contig_words["), 1)
        loops, tail = stage.split("if (pending_valid)", 1)
        self.assertNotIn("s_shared_row_contig_words[", loops)
        self.assertIn("word_index.to_uint() >= row_word_count", tail)
        self.assertIn("pending_word", tail)
        self.assertNotIn("on_chip_memory_", stage)

    def test_vec_pack_state_uses_proven_small_ranges(self):
        source = (self.root / "src/vec_alu_engine.cpp").read_text(encoding="utf-8-sig")
        self.assertIn("ap_uint<2> packed_carry_bytes", source)
        self.assertIn("ap_uint<3> packed_group_slot", source)
        self.assertIn("ap_uint<9> packed_word_idx", source)
        self.assertIn("u32_t packed_carry", source)
        self.assertNotIn("unsigned packed_group_slot", source)
        self.assertNotIn("unsigned packed_word_idx", source)

    def test_vec_and_block5_share_exact_bounded_rounding(self):
        q = (self.root / "include/npu_q.hpp").read_text(encoding="utf-8-sig")
        vec = (self.root / "src/vec_alu_engine.cpp").read_text(encoding="utf-8-sig")
        ppu = (self.root / "src/ppu.cpp").read_text(encoding="utf-8-sig")
        rounding = structure.function_body(q, "round_shift_bounded") or ""
        self.assertIn("ap_int<Width + 1>", rounding)
        self.assertIn("static_cast<i32_t>", rounding)
        self.assertNotIn("i64_t", rounding)
        self.assertIn("round_shift_bounded<40>", q)
        self.assertIn("round_shift_bounded<41>", q)
        self.assertIn("affine_i8_to_i8_bounded(",
                      structure.function_body(vec, "vec_alu_apply_affine_half"))
        block = structure.function_body(ppu, "ppu_apply_block_word")
        self.assertIn("add_i8_bounded(lhs, rhs, add_qparam)", block)
        self.assertIn("affine_i8_to_i8_bounded(", block)
        self.assertIn("#pragma HLS PIPELINE II=1", block)
        for retired in ("vec_round_shift_affine_narrow", "vec_affine_i8_to_i8_narrow",
                        "vec_aff_round_t", "vec_aff_acc_t"):
            self.assertNotIn(retired, vec)

    def test_round8_validation_runs_only_the_final_full_frame(self):
        script = (self.root / "run_csim_1011_round6.tcl").read_text(encoding="utf-8-sig")
        self.assertIn("{6 7 7r 8}", script)
        self.assertIn("if {$round in {7 7r 8}} {set tests {full}}", script)
        self.assertIn("if {$round in {7 7r 8} && $tests ne \"full\"}", script)


class GrayHlsContractTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "ESP_INT8_hls"

    def test_schedule_fields_do_not_compare_ap_int_directly_with_enums(self):
        field = r"\b(?:desc|sched)\.(?:mode|loader_class|flags)\b"
        enum = r"\b(?:WIN_MODE_|WIN_LOADER_|WINDOW_SCHED_FLAG_)\w+\b"
        comparison = rf"(?:{field}\s*(?:==|!=)\s*{enum}|{enum}\s*(?:==|!=)\s*{field})"
        for path in (self.root / "src").glob("*.cpp"):
            source = path.read_text(encoding="utf-8-sig")
            self.assertEqual(re.findall(comparison, source), [],
                             f"{path.name}: ambiguous synthesis AP-int/enum comparison")

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
        for channels in (17, 129):
            self.assertIn(f"build_staged_c{channels}_word(", win)
        self.assertIn("scheduled_c1_packed_row(", win)
        self.assertNotIn("build_staged_c1_word(", win)
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
