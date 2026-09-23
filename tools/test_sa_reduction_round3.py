from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SA_CORE = ROOT / "ESP_INT8_hls/src/sa_core.cpp"
SA_TB = ROOT / "ESP_INT8_hls/tb/sa_core_tb.cpp"


class SaReductionRound3Tests(unittest.TestCase):
    def test_single_fixed_five_level_reduction_tree(self) -> None:
        source = SA_CORE.read_text(encoding="utf-8")
        self.assertEqual(source.count("dot_product_tree_32("), 2)
        self.assertIn("ap_int<16> products[TK]", source)
        for width, count in ((17, 16), (18, 8), (19, 4), (20, 2), (21, 1)):
            self.assertIn(f"ap_int<{width}> sum_l{width - 16}[{count}]", source)
        self.assertNotIn("partial +=", source)
        self.assertIn("psum[lane] += partial", source)

    def test_array_and_stream_contracts_are_unchanged(self) -> None:
        source = SA_CORE.read_text(encoding="utf-8")
        self.assertIn("SA_ACTIVE_TM = 32", source)
        self.assertIn("SA_OUTPUT_TM = 16", source)
        self.assertIn("SA_K_TILE_II = 1", source)
        self.assertIn("#pragma HLS PIPELINE II=SA_K_TILE_II", source)
        self.assertEqual(source.count("mac_tile_active_lanes("), 2)
        self.assertNotIn("mac_tile_low_group(", source)
        self.assertNotIn("mac_tile_high_group(", source)
        self.assertNotIn("#pragma HLS ALLOCATION", source)

    def test_only_int8_products_are_explicitly_bound_to_dsps(self) -> None:
        source = SA_CORE.read_text(encoding="utf-8")
        self.assertEqual(
            source.count("#pragma HLS BIND_OP variable=product op=mul impl=dsp"),
            1,
        )
        for variable in ("sum_l1", "sum_l2", "sum_l3", "sum_l4", "sum_l5", "partial", "psum"):
            self.assertNotIn(f"BIND_OP variable={variable}", source)

    def test_focused_tb_covers_round3_k_depths_and_edges(self) -> None:
        tb = SA_TB.read_text(encoding="utf-8")
        for tiles in (1, 2, 4, 8, 37):
            self.assertIn(f"run_k_tile_case({tiles}", tb)
        self.assertIn("test_signed_extremes()", tb)
        self.assertIn("test_two_pixel_odd_tail()", tb)


if __name__ == "__main__":
    unittest.main()
