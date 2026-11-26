# Unified INT8 HLS Component

Branch: feature/p7-unified-int8. Production entry: hls_config.cfg;
top: espnet_encoder_int8_core. Synthesis/implementation remain manual.

## Module Owners

- int8_core.cpp: external interfaces, runtime validation and INIT/RUN.
- main_ctrl.cpp: PARAM EXEC scheduling, single dispatch tail.
- conv_engine.cpp: one shared Conv row DATAFLOW and one SA call site.
- win_gen.cpp: PARAM-scheduled windows; shared C1/C3 input-row reuse.
- sa_core.cpp: shared 32x32 MAC/reduction, 16-lane psum output.
- ppu.cpp: fused row consumers and local BLOCK5 finalization.
- vec_alu_engine.cpp: fixed affine/add families and row commit.
- avgpool_unit.cpp: one shared C1/C3 group engine.
- memory.cpp / scratch_mgr.cpp: one physical RAM owner and lifetime protection.
- conv_store.cpp: compact/aligned store and writeback owner.
- frame_dma.cpp: bounded native/crop-select-pack input pipeline.
- upsample_unit.cpp: shared bilinear-logits/argmax mask-output owner.
- param_dma.cpp: PARAM v4 descriptor/qparam loading and validation.

No legacy if_dec, concat, standalone BLOCK5 finalizer, task-sequence mux or
ALLOCATION directive is required. Different modes do not instantiate separate NPUs.

## Runtime Contract

See tools/geometry_contract.py and report_and_workplans/workplan_1116.md.
Eight encoder profiles share 0x530000 bytes FMBUF, including 0x380000 bytes
URAM prefix with one shared 256-bit word bus. The four-bank experiment was
removed after physical routing and packing regressed; logical addresses and
the unified layout are unchanged. Affine coefficients use eight packed 320-bit
BRAM groups, with 26 live records in a 32-record allocation, not lane-wise auto RAMs.
Descriptor caches are bounded by the finite 19-tensor/16-EXEC graph; frame resets
invalidate aliases without resetting feature RAM. Conv configuration loads one
shared descriptor/schedule/qparam set before inline mode adapters. Actual RAM binding, resource
use and read-path throughput still require csynth and implementation review.
Runtime ports: model_profile_id, src_mode, mask_mode; readable runtime_abi_id,
runtime_capability, runtime_status. Profiling outputs are unchanged.
The unified ABI/layout requires new PARAMs, a repackaged IP and a new platform.

## Verification

The default TB reads hw_artifacts/unified_int8/RUNS.CSV, not a fixed native512
artifact. Override only with ESP_INT8_CSIM_RUNTIME_ROOT pointing at another
validated unified catalog. It checks ten published cases plus a return to the
first model, logits/mask bytes, active output bounds and negative commands.
Use tools/test_unified_runtime_contract.py for host CSim and component checks.

An optional Vitis CSim runner is run_csim_unified.tcl; it never launches csynth.
Host CSim is not RTL co-simulation or a routing/resource guarantee. Sequential
stream-depth diagnostics must not be used to size DATAFLOW FIFOs.
