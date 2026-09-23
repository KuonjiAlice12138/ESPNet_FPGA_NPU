from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1] / "ESP_INT8_hls"


class ConvPostStructureTests(unittest.TestCase):
    def test_one_word_wide_requant_loop_and_common_write_tail(self):
        text = (ROOT / "src/conv_engine.cpp").read_text()
        body = text.split("void post_process_conv_row_to_buffer(", 1)[1].split(
            "static void generate_conv_window_row", 1)[0]
        self.assertIn("POST_LANES_PER_CYCLE = 16", body)
        self.assertEqual(body.count("requant_i32_to_i8("), 1)
        self.assertEqual(body.count("psum_stream.read()"), 1)
        self.assertEqual(len(re.findall(r"row_buf\[[^]]+\]\s*=", body)), 1)
        self.assertNotIn("out_lanes[TM]", body)
        self.assertNotIn("qparam.bias[q_lane]", body)
        self.assertNotIn("POST_REQUANT_GROUPS", body)

    def test_one_profile_write_per_psum_word_in_hot_loop(self):
        text = (ROOT / "src/conv_engine.cpp").read_text()
        body = text.split("void post_process_conv_row_to_buffer(", 1)[1].split(
            "static void generate_conv_window_row", 1)[0]
        loop = body.split("for (int half =", 1)[1].split(
            "\n  npu_profile_set_conv_post_state", 1)[0]
        self.assertEqual(loop.count("npu_profile_set_conv_post_state("), 1)
        self.assertEqual(body.count("npu_profile_set_conv_post_state("), 2)
        self.assertLess(loop.index("npu_profile_set_conv_post_state("),
                        loop.index("psum_stream.read()"))
        self.assertIn("PROF_CONV_POST_REQUANT_OR_WAIT_PSUM", loop)
        self.assertIn("PROF_CONV_POST_ROWBUF_WRITE", loop)
        self.assertIn("#pragma HLS PIPELINE II=1", loop)
        self.assertNotIn("#pragma HLS DEPENDENCE", body)

    def test_single_dataflow_and_unchanged_sa_and_fifo(self):
        conv = (ROOT / "src/conv_engine.cpp").read_text()
        sa = (ROOT / "src/sa_core.cpp").read_text()
        self.assertEqual(conv.count("post_process_conv_row_to_buffer("), 2)
        self.assertEqual(conv.count("systolic_array_core_row("), 2)
        row_engine = conv.split("static void shared_conv_row_engine(", 1)[1].split(
            "static void replay_conv_row_to_stream", 1)[0]
        self.assertEqual(row_engine.count("#pragma HLS DATAFLOW"), 1)
        self.assertIn("variable=psum_stream depth=8", conv)
        self.assertIn("variable=psum_stream type=fifo impl=lutram", conv)
        self.assertIn("SA_ACTIVE_TM = 32", sa)
        self.assertIn("SA_K_TILE_II = 1", sa)


if __name__ == "__main__":
    unittest.main()
