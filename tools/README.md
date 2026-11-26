# ESP INT8 Tools

Current branch: feature/p7-unified-int8. One IP and one UART App select the encoder,
source crop/resize path and output grid at runtime. PARAM v4 alone is not sufficient:
only p7-unified-layout-v1 artifacts with runtime ABI 0x11160001 match this IP.
Training/QAT tools remain in D:/ESPNet; this integration does not retrain models.

## Publish and Validate

The selected release is hw_artifacts/unified_int8. RUNS.CSV registers ten cases:
G240B2, R256B2, R256B20, G512B2, and G2S1M0 through G2S3M1.
The six G2S cases use frozen real high-resolution grayscale inputs and the gray240
model, with either a 240 mask or a mask covering the original center ROI.
MODELS.JSN distinguishes hardware capability from available trained models.
Missing trained models, including the absent maximum-RGB deployment artifact,
are not replaced by geometry-adapted structural fixtures.

Each case contains MODEL.JSN, PARAM.BIN, INPUTQ.BIN (source HWC INT8),
ENCODER.BIN (selected encoder HWC INT8), LOGITS.BIN (HWC INT8),
MASK.BIN (bilinear logits then argmax) and TARGET.BIN when registered GT exists.
Hashes bind input, qparams, logits, preprocessing, geometry, ROI and GT view.
All SD paths use 8.3 components. OUT.BIN is reserved for fresh board output.

Run from D:/ESP_INT8, with D:/ESPNet/.venv/Scripts/python.exe -B:

- export_int8_hw_blob.py --publish-runtime-catalog --out-dir NEW_DIRECTORY
  publishes the four frozen local models and six real lite cases atomically.
- export_int8_hw_blob.py --runtime-from FROZEN_ARTIFACT --out-dir NEW_DIRECTORY
  readdresses an accepted native artifact without changing weights/qparams.
  Lite modes also require --src-mode, --mask-mode, --source-input,
  --frozen-logits and the corresponding --source-contract.
- export_int8_hw_blob.py --artifact-dir MODEL_EXPORT --out-dir NEW_DIRECTORY
  compiles a model-side export and publishes the unified layout. Legacy layouts
  require explicit --legacy-layout and must never be used with the unified IP.
- export_int8_hw_blob.py --emit-app-contract ESP_INT8_app/src/runtime_profiles.h
  regenerates the C profile/IO table from the same Python contract as HLS.
- export_int8_hw_blob.py --audit-runtime-memory checks eight geometries times
  two class counts, lifetimes, B2 protection and the conservative RAM budget.
- hw_param_replay.py --runtime-artifact hw_artifacts/unified_int8/G2S1M1
  validates the manifest, executes PARAM and checks logits and mask.

The old frozen artifact directories are publication inputs and historical
references, not inputs to the new IP. Publication refuses to overwrite them.
No binary artifacts or temporary outputs should be committed.

## Evaluate and Export Datasets

eval_single_hw_outputs.py accepts a unified case through --artifact-dir and
validates MODEL.JSN before evaluating --fullres-mask/--lowres-logits. Use the registered
TARGET.BIN or explicitly registered original GT; do not substitute a legacy
cache, guess shapes from byte counts, or mix NCHW model arrays with HWC bins.
PA/mIoU reports must identify model, source/ROI/output grid, and sample count.

export_val_hw_dataset.py --runtime-artifact CASE --source-manifest SOURCES.JSN
--out-dir NEW_DIRECTORY exports explicitly model-bound, prequantized validation
sources. Its --help and module docstring define the source-manifest schema.
It preserves the manifest's normalization,
quantization and source/GT transforms. It does not train, calibrate or silently
change model geometry. Old dataset and evaluation entry points remain explicit
historical reproduction paths only.

For the unified VAL500 App, the same exporter also packages complete datasets:

    python -B tools/export_val_hw_dataset.py --board-suite SPEC.JSON --out-dir NEW_BOARD_DIR

SPEC.JSON uses `{"format":"P7_UNIFIED_VAL500_SUITE_V1","datasets":[{"key":"G240B2","dataset":"DATASET/manifest.json"}]}`.
Each dataset must already be `P7_UNIFIED_INT8_VAL_DATASET_V1`, bound to its exact
MODEL.JSN/quantization/preprocessing, with indices 0..499 and the same 500 unique
image names/order across modes. Packaging does not run training or replay. It
preflights input/GT identities, class IDs and output geometry; emits short
`CASE/Ixxxx.BIN`, `Txxxx.BIN`, PARAM/MODEL, `SAMPLES.CSV`, provenance and root
`VALRUNS.CSV`; never stages golden masks as board `Oxxxx.BIN` results. App autorun
starts only when VALRUNS.CSV exists. Files stay in the new host directory, not E:.
`SUITE.JSN` reports staged/output space and modes not included; hardware capability
does not imply trained-model or 500-image dataset availability. See
`ESP_INT8_app/src/README.txt` for commands and PA/mIoU/timing conventions.

The ten-case 2026-11-22 dataset producer is model-side:
`D:/ESPNet/unified_val500/prepare_val500.py`. It freezes the accepted input
quantization and GT transforms, aligns the RGB/gray caches by the same 500 image
IDs, and checks all ten published input/GT/selected-encoder references exactly.
It emits input/GT-only unified datasets, explicitly without software predictions,
reference metrics or retraining. Package its SUITE_SPEC.JSON with --board-suite.
Current host SD package: `hw_artifacts/unified_val500_1122/board`; its RUNINFO.JSN
binds the user-built ELF, platform XSA and producer hashes. All 10 published
cases passed board validation on 500 images each. See [workplan_1116](../report_and_workplans/workplan_1116.md)
for accuracy, timing and known performance limits. The other 30 hardware
combinations are not claimed as tested.

## Verification

Regular contracts, components, App host tests and source ownership:

    python -B -m unittest discover -s tools -p "test_*.py"
    python -B tools/check_p6_hls_structure.py

Final full-network host CSim using the production default TB and all published
cases in one process, followed by A->B->A return to the first model:

    $env:ESP_INT8_RUNTIME_RELEASE_CHECK = '1'
    python -B -m unittest tools.test_unified_runtime_contract.RuntimePublishingTests.test_default_tb_all_published_models_in_one_process -v

Maximum RGB20 and RGB-lite structural fixtures (not trained deployment models):

    $env:ESP_INT8_FULL_RUNTIME_CASES = '5,6'
    python -B -m unittest tools.test_unified_runtime_contract.RuntimeNetworkTests -v

TemporaryDirectory automatically removes host binaries, probe files and dumps.
These are C simulations linked against the actual HLS sources and vendor
ap_int/hls::stream headers, not RTL co-simulation. Sequential host stream peaks
are not hardware FIFO-depth requirements. FIFO token/backpressure tests cover
the bounded input DAG but cannot prove Vivado/HLS RTL scheduling.

The historical checker filename is retained. It now checks current unified
owners and ports; --report-root PATH additionally flags possible csynth clones.
Actual resource use, RAM binding, II, clone count, timing and FIFO liveness must
still be checked after manual csynth/implementation.

audit_unified_checkpoint.tcl opens a supplied placed/failed DCP read-only and
reports route status, critical paths, high fanout and utilization. Run Vivado in
a temporary working directory; it does not synthesize, place, route or save the DCP.

Other retained tools cover integer math, legacy compiler regression, report
audits, EXEC profiling and useful-MAC analysis. WIN/SA/POST overlap; never sum
them as separate serial phase times. No csynth, implementation, packaging,
App build, SD copy, training or Git operation is launched by these checks.
