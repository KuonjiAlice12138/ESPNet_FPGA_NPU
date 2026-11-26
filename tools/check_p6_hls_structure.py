#!/usr/bin/env python3
"""Source-level singleton/runtime gates. Historical filename, current unified IP.
These are necessary structural checks, not proof of binding or routability.
Use --report-root after manual csynth to flag possible replicated hierarchies.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HLS_DIR=ROOT/"ESP_INT8_hls"
SRC_DIR=HLS_DIR/"src"
INC_DIR=HLS_DIR/"include"

def function_body(source: str, name: str) -> str | None:
    match = re.search(
        rf"(?:static\s+)?[\w:<>\s*&]+\s+{re.escape(name)}\s*\([^;]*?\)\s*\{{",
        source,
        flags=re.DOTALL,
    )
    if not match:
        return None
    open_pos = source.find("{", match.start())
    return braced_body(source, open_pos)


def braced_body(source: str, open_pos: int) -> str | None:
    depth = 0
    for pos in range(open_pos, len(source)):
        ch = source[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[open_pos + 1 : pos]
    return None


def count_call_sites(source: str, token: str) -> int:
    source=re.sub(r'/\*.*?\*/|//[^\n]*','',source,flags=re.S)
    return len(re.findall(rf"\b{re.escape(token)}\s*\(", source))

def load_sources() -> dict[str,str]:
    source={path.name:path.read_text(encoding="utf-8-sig") for path in SRC_DIR.glob("*.cpp")}
    source.update({path.name:path.read_text(encoding="utf-8-sig") for path in INC_DIR.glob("*.hpp")})
    for name,path in (("app_main.c",ROOT/"ESP_INT8_app/src/main.c"),
                      ("app_hal.c",ROOT/"ESP_INT8_app/src/hal/int8_npu.c"),
                      ("hls_config.cfg",HLS_DIR/"hls_config.cfg")):
        source[name]=path.read_text(encoding="utf-8-sig")
    return source

def check_unified_sources(source: dict[str,str]) -> list[str]:
    errors=[]
    def require(condition,message):
        if not condition: errors.append(message)
    def body(file,name):
        value=function_body(source[file],name)
        require(value is not None,f"{file}: missing {name}")
        return value or ""
    def one(file,owner,callee):
        require(count_call_sites(body(file,owner),callee)==1,
                f"{file}: {owner} must have exactly one {callee} call site")
    for filename,text in source.items():
        if not filename.endswith((".cpp",".hpp")): continue
        require(not re.search(r"#pragma\s+HLS\s+ALLOCATION\b",text),
                f"{filename}: do not use ALLOCATION to hide source-level replication")
        for token in ("ESP_INT8_ENABLE_COLD_RMW_FALLBACK","run_conv_task_sequence",
                      "s_shared_conv_task_seq","ppu_read_block5_scratch_segment",
                      "legacy_window_generator","concat_unit","param_blob_v1"):
            require(token not in text,f"{filename}: retired path {token}")
    cpp={name for name in source if name.endswith(".cpp") and not name.startswith("app_")}
    expected={name+".cpp" for name in ("int8_core","main_ctrl","conv_engine","ppu","vec_alu_engine",
        "memory","scratch_mgr","frame_dma","param_dma","conv_store","upsample_unit","win_gen","sa_core","avgpool_unit")}
    require(cpp==expected,"Unexpected/missing hardware CPP module owner")
    configured=set(re.findall(r"^syn.file=src/([\w.]+)$",source["hls_config.cfg"],re.M))
    require(configured==expected,"Production cfg source list differs from module owners")
    require("syn.top=espnet_encoder_int8_core" in source["hls_config.cfg"],"Wrong synthesis top")
    require("tb.file=tb/top_golden_sample_tb.cpp" in source["hls_config.cfg"],"Wrong default TB")
    one("int8_core.cpp","core_mode_run","main_ctrl_run")
    one("main_ctrl.cpp","main_ctrl_run","execute_issue")
    one("conv_engine.cpp","run_conv_rows_task","shared_conv_row_engine")
    one("conv_engine.cpp","shared_conv_row_engine","systolic_array_core_row")
    one("conv_engine.cpp","generate_conv_window_row","scheduled_window_generator_row")
    require(count_call_sites(source["conv_engine.cpp"],"systolic_array_core_row")==2,
            "Conv/SA must have one declaration and one physical call site")
    require("#pragma HLS DATAFLOW" in body("conv_engine.cpp","shared_conv_row_engine"),
            "Conv row DATAFLOW boundary lost")
    for variable in ("s_shared_conv_row_buf","s_shared_weight_buf"):
        require(len(re.findall(r"\bstatic\s+[\w:<>]+\s+"+variable+r"\s*\[",
                               source["conv_engine.cpp"]))==1,f"Duplicate/missing {variable} owner")
    for array in ("s_fmbuf_uram","s_fmbuf_bram"):
        require(len(re.findall(r"\bstatic\s+\w+\s+"+array+r"\s*\[",source["memory.cpp"]))==1,
                f"Duplicate/missing {array}")
        require(f"RESET variable={array} off" in source["memory.cpp"],f"{array}: global reset expanded")
    require("s_fmbuf_uram[FMBUF_URAM_AXI_WORDS]" in source["memory.cpp"] and
            "FMBUF_URAM_BANKS" not in source["npu_config.hpp"] and
            "ARRAY_PARTITION variable=s_fmbuf_uram" not in source["memory.cpp"],
            "Do not restore the failed four-bank global return fabric")
    for callee in ("param_dma_get_conv_exec_desc", "param_dma_get_window_sched",
                   "param_dma_get_conv_qparam", "conv_cfg_from_exec_desc"):
        one("conv_engine.cpp", "build_conv_task_from_issue", callee)
    for adapter in ("configure_normal_consumer", "configure_block5_consumer"):
        adapter_body = body("conv_engine.cpp", adapter)
        require("#pragma HLS INLINE\n" in adapter_body and "#pragma HLS INLINE off" not in adapter_body,
                f"{adapter}: must not become another registered full-task builder")
    for token in ("build_normal_conv_task", "build_block5_branch_task", "block5_prev_branch", "result.conv_desc"):
        require(token not in source["conv_engine.cpp"], f"Retired task payload/path: {token}")
    for pragma in ("ARRAY_PARTITION variable=s_affine_qparam_words complete dim=1",
                   "BIND_STORAGE variable=s_affine_qparam_words type=ram_2p impl=bram",
                   "RESET variable=s_affine_qparam_words off"):
        require(pragma in source["param_dma.cpp"], "Packed affine BRAM contract missing: " + pragma)
    require("static affine_qparam_t s_affine_qparam[" not in source["param_dma.cpp"],
            "Old per-lane auto-RAM affine cache remains beside packed cache")
    require("s_global_alias_epoch" not in source["scratch_mgr.cpp"] and
            "s_global_alias_valid" in source["scratch_mgr.cpp"],
            "Global aliases must use frame-cleared validity, not a wide epoch/tag fabric")
    require("on_chip_memory_write_input_axi_word" in source["frame_dma.cpp"] and
            function_body(source["memory.cpp"],"on_chip_memory_write_input_axi_word") is not None,
            "Input DMA write helper removed or bypassed")
    require("runtime_select_profile" in source["memory.cpp"] and "runtime_select_io" in source["memory.cpp"],
            "Runtime configuration must remain with the memory owner")
    for function in ("reset_window_state","first_load_packed_source_row","scheduled_first_packed_row"):
        require(function_body(source["win_gen.cpp"],function) is not None,f"Missing shared first-conv path {function}")
    for mode in ("WIN_MODE_3X3_STAGED_C19","WIN_MODE_3X3_STAGED_C131"):
        require(mode in source["win_gen.cpp"],f"RGB schedule mode missing: {mode}")
    one("avgpool_unit.cpp","avgpool_unit_checked","avgpool_group_engine")
    sa=source["sa_core.cpp"]
    for token in ("SA_ACTIVE_TM = 32","SA_OUTPUT_TM = 16","SA_K_TILE_II = 1",
                  "PIPELINE II=SA_K_TILE_II","ap_int<21> sum_l5[1]"):
        require(token in sa,f"SA performance/shape contract missing: {token}")
    require(count_call_sites(sa,"mac_tile_active_lanes")==2,"SA MAC owner replicated")
    require(count_call_sites(sa,"dot_product_tree_32")==2,"SA reduction owner replicated")
    require("POST_LANES_PER_CYCLE = 16" in source["conv_engine.cpp"],"POST16 changed")
    require("ap_uint<11> last_word" in source["conv_store.cpp"],"1048-word span cursor truncated")
    require("for (ap_uint<11> word = 0; word < ROW_CONTIG_MAX_WORDS;" in source["vec_alu_engine.cpp"],
            "1048-word commit cursor truncated")
    for function in ("reset_scratch_state","resolve_block5_scratch_descs"):
        require(function_body(source["scratch_mgr.cpp"],function) is not None,f"Scratch owner missing {function}")
    require("s_b2_saved" in source["scratch_mgr.cpp"],"B2 residual overwrite protection missing")
    dma=body("frame_dma.cpp","frame_dma_pipeline")
    require("#pragma HLS DATAFLOW" in dma,"Input pipeline no longer DATAFLOW")
    for variable,depth in (("source_words",16),("packed_words",4)):
        require(f"variable={variable} depth={depth}" in dma,f"Input FIFO {variable} depth changed")
    require("return " not in dma,"Nested DATAFLOW function must return void")
    one("upsample_unit.cpp","upsample_fused_consume_logits_row","emit_output_row")
    for port in ("model_profile_id","src_mode","mask_mode","runtime_abi_id","runtime_status","runtime_capability"):
        require(f"s_axilite port={port} bundle=control" in source["int8_core.cpp"],f"Missing AXI-Lite port {port}")
    for port in ("prof_stage_id","prof_active","prof_pc","prof_issue_kind",
                 "prof_conv_win_state","prof_conv_sa_state","prof_conv_post_state"):
        require(f"port={port} register" in source["int8_core.cpp"],f"Missing profiling port {port}")
    require("RUNTIME_MODE_MASK" not in source["int8_core.cpp"],"Legacy upper-bit mode alias remains")
    app=source["app_main.c"]
    for token in ("runtime_parse_case(","int8_npu_select_runtime(","stage_counter_clear();",
                  "stage_counter_enable(1U);","stage_counter_freeze(1U);","stage_counter_dump_csv();",
                  "runtime_crc32(","timer_frequency(","profile_valid(","npu.blocked"):
        require(token in app,f"App runtime delivery check missing {token}")
    for token in ("ENABLE_EXEC_PREFIX","ENABLE_DUAL_SINGLE","INT8_APP_INPUT_H"):
        require(token not in app,f"Retired compile-time App selection remains: {token}")
    hal=source["app_hal.c"]
    for token in ("MODEL_PROFILE_ID_DATA","SRC_MODE_DATA","MASK_MODE_DATA","RUNTIME_ABI_ID_DATA",
                  "RUNTIME_STATUS_DATA","RUNTIME_CAPABILITY_DATA"):
        require("XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_"+token in hal,f"HAL missing generated address {token}")
    return errors

def check_report(root: Path) -> list[str]:
    report=root/"csynth.rpt"
    if not report.is_file(): return [f"Missing {report}"]
    text=report.read_text(encoding="utf-8",errors="replace")
    errors=[]
    for name in ("shared_conv_row_engine","scheduled_window_generator_row","systolic_array_core_row"):
        if name not in text: errors.append(f"Cannot verify singleton {name}: absent from hierarchy")
        clones=set(re.findall(r"\b"+re.escape(name)+r"_\d+\b",text))
        if clones: errors.append(f"Possible cloned hierarchy {name}: {sorted(clones)}; inspect actual instances")
    return errors

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--report-root",type=Path)
    args=parser.parse_args()
    errors=check_unified_sources(load_sources())
    if args.report_root: errors+=check_report(args.report_root)
    for error in errors: print("[MC-CHECK] FAIL:",error)
    if errors: return 1
    print("[MC-CHECK] PASS unified runtime/source ownership; source checks do not prove RTL binding or timing")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
