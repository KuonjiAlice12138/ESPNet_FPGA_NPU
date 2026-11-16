"""Native512 grayscale compiler/replay contracts; no synthesis is launched."""

from dataclasses import fields, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
import re
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import export_int8_hw_blob as compiler
from tools import hw_param_replay as replay
from tools.geometry_contract import DeploymentGeometry


GEOMETRY = DeploymentGeometry(512, 512, 8, input_channels=1)
MODEL_ROOT = Path(r'D:\ESPNet\resolution_h240w240_gray\quantized_artifacts_binary2_int8_hw')


class NativeCompilerTests(unittest.TestCase):
    def tearDown(self):
        compiler.activate_deployment_plans(geometry=compiler.DEFAULT_GEOMETRY)

    def test_native_contract_accepts_frozen_gray_metadata(self):
        manifest = json.loads((MODEL_ROOT/'manifest.json').read_text(encoding='utf-8'))
        manifest['geometry'] = GEOMETRY.to_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            contract = compiler.load_deployment_contract(root)
            self.assertEqual(contract.geometry.to_manifest(), GEOMETRY.to_manifest())
            manifest['deployment']['class_count'] = 20
            manifest['deployment']['profile_name'] = 'cityscapes20'
            (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                compiler.load_deployment_contract(root)

    def test_native_memory_has_physical_rows_and_pool_tail(self):
        layout = compiler.build_memory_layout(GEOMETRY)
        self.assertEqual(layout.fmbuf_bytes, 0x378000)
        self.assertEqual(layout.uram_bytes, 0x358000)
        self.assertEqual(layout.residual_base, 0x208000)
        self.assertEqual(layout.c1_scratch_base, 0x318000)
        self.assertEqual(layout.pool1_bytes, 256*256)
        compiler.activate_deployment_plans(geometry=GEOMETRY)
        self.assertEqual(compiler.TENSORS[9].row_stride_bytes, 128*129)
        self.assertEqual(compiler.TENSORS[9].row_stride_bytes//32, 516)
        audit = compiler.build_memory_lifetime_audit(compiler.build_uops(geometry=GEOMETRY), geometry=GEOMETRY)
        self.assertEqual(audit['physical_layout']['estimated_uram_blocks_256b'], 108)

    def test_block5_schedule_covers_two_groups_and_rejects_mismatched_geometry(self):
        compiler.activate_deployment_plans(geometry=GEOMETRY)
        uops = compiler.build_uops(geometry=GEOMETRY)
        _, _, schedule_ids, _ = compiler.build_window_schedule_sections(uops)
        conv_uops = [u for u in uops if u.opcode == compiler.UOP_CONV]
        offsets = {u.param_id: u.param_id*2048 for u in conv_uops}
        weights = [dict(param_id=u.param_id, k_tiles=(u.in_c*u.kernel*u.kernel+31)//32,
                        packed_words=((u.in_c*u.kernel*u.kernel+31)//32)*u.out_c) for u in conv_uops]
        convs, _, _, _, schedules, _ = compiler.build_exec_plan_sections(uops, schedule_ids, offsets, weights)
        first = schedules[0]
        compiler.validate_block5_schedule(first, convs)
        self.assertEqual((first.out_h, first.out_w, first.row_group_h), (128, 128, 64))
        for bad in (replace(first, row_group_h=0), replace(first, out_h=64), replace(first, out_w=60)):
            with self.assertRaises(ValueError):
                compiler.validate_block5_schedule(bad, convs)

    def test_native_first_conv_uses_sixteen_word_keep1(self):
        desc, _ = compiler.make_window_pack_schedule(
            1, 3, source_w=512, stride=2, dilation=1, padding=1,
            out_w=256, flags=compiler.WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)
        self.assertEqual(desc.mode, compiler.WIN_MODE_3X3_STAGED_C1)
        self.assertEqual(desc.reserved[9], 2 | (16 << 2))
        desc = replay.WindowSchedDesc(**{f.name: getattr(desc, f.name) for f in fields(replay.WindowSchedDesc)})
        conv = SimpleNamespace(in_h=512, in_w=512, in_c=1, out_c=16, kernel=3,
                               stride=2, dilation=1, padding=1, k_tiles=1)
        source = replay.TensorDesc(h=512, w=512, c=1, reserved0=1)
        self.assertTrue(replay.window_row_reuse_contract_valid(desc, conv, source))
        words = list(desc.reserved)
        words[9] = 2 | (8 << 2)
        self.assertFalse(replay.window_row_reuse_contract_valid(replace(desc, reserved=words), conv, source))

    def test_existing_narrow_transfer_protocols_balance_at_native_widths(self):
        for channels, width, paired, stride in ((12, 128, True, 1), (17, 256, True, 2), (25, 64, False, 1)):
            for dilation in (1, 2, 4, 8, 16):
                if channels == 17 and dilation != 1:
                    continue
                desc, _ = compiler.make_window_pack_schedule(
                    channels, 3, source_w=width, stride=stride, dilation=dilation, padding=dilation,
                    out_w=width//stride, flags=compiler.WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2 if paired else 0)
                audit = compiler.audit_window_transfer_contract(desc)
                self.assertEqual(audit['produced_words'], audit['consumed_words'])

    def test_replay_memory_supports_operation_derived_capacity_and_checks_bounds(self):
        memory = replay.FeatureMemory(0x21D000)
        desc = replay.TensorDesc(base_offset=0x21CFE0, h=1, w=1, c=32)
        values = np.arange(32, dtype=np.int8)
        memory.store_tile(desc, 0, 0, 0, 32, values)
        np.testing.assert_array_equal(memory.load_tile(desc, 0, 0, 0, 32), values)
        with self.assertRaises(ValueError):
            memory.store_tile(replace(desc, base_offset=0x21CFE1), 0, 0, 0, 32, values)

    def test_hls_constants_match_generated_layout_and_tight_row_bound(self):
        root = Path(__file__).resolve().parents[1]
        config = (root/'ESP_INT8_hls/include/npu_config.hpp').read_text(encoding='utf-8-sig')
        values = {}
        for name, expression in re.findall(r'constexpr int (\w+)\s*=\s*([^;]+);', config):
            values[name] = eval(expression.replace('/', '//'), {'__builtins__': {}}, values)
        layout = compiler.build_memory_layout(GEOMETRY)
        for name, expected in dict(INPUT_FRAME_H=512, INPUT_FRAME_W=512, INPUT_FRAME_C=1,
                ENCODER_OUT_H=64, ENCODER_OUT_W=64, INPUT_FRAME_AXI_WORDS=8192, OUTPUT_FRAME_AXI_WORDS=8192,
                FMBUF_BYTES=layout.fmbuf_bytes, FMBUF_URAM_BYTES=layout.uram_bytes,
                FMBUF_BLOCK5_BASE=layout.block5_base, FMBUF_B1_BASE=layout.residual_base,
                FMBUF_L2_SCRATCH_BASE=layout.c1_scratch_base, FMBUF_POOL1_BASE=layout.pool1_base,
                FMBUF_POOL_TMP_BASE=layout.pool_tmp_base, FMBUF_L2_SCRATCH_SLOT_BYTES=layout.l2_scratch_slot_bytes,
                FMBUF_L30_SCRATCH_SLOT_BYTES=layout.l3_scratch_slot_bytes,
                ROW_CONTIG_MAX_WORDS=516, BRAM_SCR0_BYTES=layout.pool1_bytes, BRAM_SCR1_BYTES=layout.pool2_bytes).items():
            self.assertEqual(values[name], expected, name)

    def test_top_is_explicit_before_all_interface_pragmas(self):
        root = Path(__file__).resolve().parents[1]/'ESP_INT8_hls'
        core = (root/'src/int8_core.cpp').read_text(encoding='utf-8-sig')
        cfg = (root/'hls_config.cfg').read_text(encoding='utf-8-sig')
        top_name = re.search(r'^syn.top=(\w+)$', cfg, re.MULTILINE).group(1)
        body = core.split(f'void {top_name}(', 1)[1].split('{', 1)[1]
        pragma = f'#pragma HLS TOP name={top_name}'
        self.assertEqual(body.count(pragma), 1, 'Missing explicit top: implicit directive injection can lose AXI interfaces')
        self.assertLess(body.index(pragma), body.index('#pragma HLS INTERFACE'))

    def test_final_blob_and_row_group_negative_case(self):
        root = Path(__file__).resolve().parents[1]/'hw_artifacts/binary2_int8_h512w512_gray_finetuned_v4'
        audit = json.loads((root/'param_audit.json').read_text(encoding='utf-8'))
        compiler.parse_and_check_blob_v4(root/'PARAM.BIN', audit)
        data = bytearray((root/'PARAM.BIN').read_bytes())
        import struct
        manifest = json.loads((root/'export_manifest.json').read_text(encoding='utf-8'))
        block_offset = manifest['section_offsets']['block5_sched_desc']
        self.assertEqual(struct.unpack_from('<I', data, 120)[0], block_offset)
        data[block_offset+20:block_offset+22] = b'\0\0'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.bin'
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'BLOCK5'):
                compiler.parse_and_check_blob_v4(path, audit)


if __name__ == '__main__':
    unittest.main()
