import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HLS = ROOT / "ESP_INT8_hls"
sys.path.insert(0, str(ROOT / "tools"))

import export_int8_hw_blob as exporter
from geometry_contract import DEFAULT_GEOMETRY


def read(relative: str) -> str:
    return (HLS / relative).read_text(encoding="utf-8")


def constexpr_hex(source: str, name: str) -> int:
    match = re.search(rf"constexpr\s+int\s+{name}\s*=\s*(0x[0-9A-Fa-f]+)\s*;", source)
    if match is None:
        raise AssertionError(f"missing literal hexadecimal constexpr {name}")
    return int(match.group(1), 16)


class Round3RMemoryLayoutContractTest(unittest.TestCase):
    def test_h256_layout_is_compact_and_matches_hls(self) -> None:
        layout = exporter.build_memory_layout(DEFAULT_GEOMETRY)
        config = read("include/npu_config.hpp")

        self.assertEqual(layout.fmbuf_bytes, 0x1F0000)
        self.assertEqual(layout.uram_bytes, 0x1C0000)
        self.assertEqual(layout.bram_bytes, 0x30000)
        self.assertEqual(layout.uram_bytes // 0x8000, 56)
        self.assertLess(layout.fmbuf_bytes, 0x598000)
        self.assertEqual(constexpr_hex(config, "FMBUF_BYTES"), layout.fmbuf_bytes)
        self.assertEqual(constexpr_hex(config, "FMBUF_URAM_BYTES"), layout.uram_bytes)
        self.assertEqual(constexpr_hex(config, "FMBUF_POOL1_BASE"), layout.pool1_base)
        self.assertEqual(constexpr_hex(config, "FMBUF_POOL_TMP_BASE"), layout.pool_tmp_base)

    def test_l3b0_c1_source_is_disjoint_from_live_block5_workspace(self) -> None:
        layout = exporter.build_memory_layout(DEFAULT_GEOMETRY)
        c1_bytes = layout.l3_scratch_slot_bytes
        block5_bytes = layout.l3_block5_bytes

        self.assertEqual(exporter.FMBUF_L3B0_SCRATCH_BASE, layout.c1_scratch_base)
        self.assertTrue(
            layout.c1_scratch_base + c1_bytes <= layout.block5_base
            or layout.block5_base + block5_bytes <= layout.c1_scratch_base,
            "U54 C1 source must remain live while U55-U68 use BLOCK5 workspace",
        )
        config = read("include/npu_config.hpp")
        self.assertEqual(
            constexpr_hex(config, "FMBUF_L3B0_SCRATCH_BASE"),
            layout.c1_scratch_base,
        )

    def test_complete_lifetime_audit_covers_transient_storage(self) -> None:
        exporter.activate_deployment_plans(2, geometry=DEFAULT_GEOMETRY)
        audit = exporter.build_memory_lifetime_audit(
            exporter.build_uops(2, geometry=DEFAULT_GEOMETRY),
            geometry=DEFAULT_GEOMETRY,
        )

        self.assertEqual(audit["format"], "ESP_INT8_MEMORY_LIFETIME_AUDIT_V2")
        self.assertTrue(audit["hard_checks"]["all_regions_in_bounds"])
        self.assertTrue(audit["hard_checks"]["all_overlaps_lifetime_safe"])
        self.assertTrue(audit["hard_checks"]["b2_backup_retired"])
        self.assertTrue(audit["hard_checks"]["l3b0_c1_disjoint_from_block5"])
        self.assertEqual(audit["physical_layout"]["high_water_mark"], 0x1F0000)
        names = {region["name"] for region in audit["transient_regions"]}
        self.assertEqual(
            names,
            {
                "L2_BRANCH_SCRATCH",
                "L3_BRANCH_SCRATCH",
                "BLOCK5_ROW_GROUP",
                "C1_SCRATCH",
                "B2_SRC1_BACKUP",
                "POOL1_OR_POOL2",
                "POOL_TMP",
            },
        )

    def test_pool_writer_is_bram_only_and_backup_deadlogic_is_removed(self) -> None:
        avgpool = read("src/avgpool_unit.cpp")
        memory = read("src/memory.cpp")
        scratch = read("src/scratch_mgr.cpp")
        vec = read("src/vec_alu_engine.cpp")

        self.assertIn("on_chip_memory_write_pool_bram_word", avgpool)
        self.assertIn("on_chip_memory_read_pool_bram_word", avgpool)
        self.assertIn("write_fmbuf_bram_word", memory)
        self.assertNotIn("on_chip_memory_write_fmbuf_abs_word", avgpool)
        self.assertNotIn("copy_src1_row_to_b2_backup", scratch)
        self.assertNotIn("read_b2_backup_src1_tile", scratch)
        self.assertNotIn("s_shared_b2_src1_saved", vec)

    def test_generated_audit_can_be_serialized(self) -> None:
        exporter.activate_deployment_plans(20, geometry=DEFAULT_GEOMETRY)
        audit = exporter.build_memory_lifetime_audit(
            exporter.build_uops(20, geometry=DEFAULT_GEOMETRY),
            geometry=DEFAULT_GEOMETRY,
        )
        json.dumps(audit)

    def test_param_audit_propagates_compact_memory_hard_gates(self) -> None:
        source = (ROOT / "tools" / "export_int8_hw_blob.py").read_text(encoding="utf-8")
        for check_name in (
            "memory_regions_in_bounds",
            "memory_overlaps_lifetime_safe",
            "b2_backup_retired",
            "l3b0_c1_disjoint_from_block5",
        ):
            self.assertIn(f'"{check_name}"', source)
        self.assertIn('"p7_h256_compact_fmbuf_v1_20260909"', source)


if __name__ == "__main__":
    unittest.main()
