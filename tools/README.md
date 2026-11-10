# ESP INT8 Tools

Current source branch: `feature/h240-mask512`, binary2 INT8, grayscale
240x240 encoder input, 30x30 logits, PARAM v4. `workplan_1104` extends source
mode 3 to a tightly packed 512x512 square and outputs a direct 512x512 mask.
External signed-INT8 inputs retain native240, 256x512 and 512x1024 central-ROI
semantics. Output size is compile-time fixed, with a 240 regression configuration.
The initial 512-mask release passed csynth, 100 MHz implementation and one mode-3
board test: 54.12012 ms, 0/262144 mask mismatches. Repeated frames/mode switching
and full-validation-set board testing remain pending. Do not use the old
240-output platform with the new App.
Model training and hardware-constrained QAT tools live in `D:/ESPNet`.

## Artifact Contract

Round 6/7/8 HLS uses `hw_artifacts/binary2_int8_h240w240_gray_r6_v4`.
The initial board release remains `hw_artifacts/binary2_int8_h240w240_gray_v4`;
its PARAM must not be used with Round 6/7/8 HLS. The v4 record layout is unchanged,
but schedule word 9 now explicitly selects C1 packed-row reuse and narrow-channel
column transfer. Round 6 INIT rejects the retired narrow transfer protocol.
Weights, qparams and encoder numerics remain identical to the baseline.
The artifact's native `INPUTQ.BIN` and historical 240 mask are each 57,600 bytes;
the new 262144-byte 512 mask uses the separate output-restore reference set.
Low-resolution HWC INT8 logits are 1,800 bytes. The manifest records grayscale
normalization, center-square crop, checkpoint/cache hashes and tensor geometry.
Intermediate compact rows carry an explicit 32-byte row-padding flag; input and
pool tensors remain tightly packed.

The Python tools can still inspect the preserved H256W512 binary2/cityscapes20
releases. Those artifacts are **not compatible with the current H240/C1 HLS IP**,
despite both using PARAM v4. The previous board release used
`platform_h240_resize_1025`, `I512Q.BIN` / `O512Q.BIN` and source mode 2.
The updated App selects `platform_1109` and mode 3, with
`I512S.BIN` / `O512S.BIN` (262144 bytes each). New hardware implementation and
platform binding are verified; App build/launch and SD preparation remain manual.
Round 8 reuses the Round 6 artifact without re-exporting or changing golden data.
Rebuild App manually after aligning its platform configuration. SD boot also requires a newly packaged
`BOOT.BIN`; `ESP_INT8_app/src/boot_h240.bif` fixes the platform/bitstream/ELF paths.

For the new geometry, run `ESP_INT8_hls/run_csim_input_resize.tcl` through
`vitis-run --mode hls --tcl`. `ESP_INT8_INPUT_RESIZE_TEST=unit` runs all four
input layouts locally, `mask240` runs the old output regression without a full
network inference, and `full` runs the only final top mode-3 inference with
frozen `output_restore_1104/selected_reference` input/logits/mask and guards.
The runner does not alter the production cfg or start csynth.

## Compile and Validate

- `export_int8_hw_blob.py`: compile PARAM v4, schedules and memory/fusion audits.
- `hw_param_replay.py`: execute PARAM using HLS-compatible integer arithmetic.
- `hw_int8_math.py`: arithmetic shared by compiler and replay.
- `geometry_contract.py`: manifest-driven input channels and geometry.
- `export_exec_prefix_params.py`: generate EXEC-prefix profiling PARAM files.
- `export_val_hw_dataset.py`: export validation inputs using the QAT loader.

Run from `D:/ESP_INT8`:

```powershell
& 'D:/ESPNet/.venv/Scripts/python.exe' tools/export_int8_hw_blob.py `
  --recompile-window-reuse-from hw_artifacts/binary2_int8_h240w240_gray_v4 `
  --out-dir hw_artifacts/binary2_int8_h240w240_gray_r6_v4

& 'D:/ESPNet/.venv/Scripts/python.exe' tools/hw_param_replay.py `
  --param hw_artifacts/binary2_int8_h240w240_gray_r6_v4/PARAM.BIN `
  --input hw_artifacts/binary2_int8_h240w240_gray_r6_v4/INPUTQ.BIN `
  --output hw_artifacts/binary2_int8_h240w240_gray_r6_v4/replay_check.bin
```

Schedule-only export changes only schedule word 9, reruns replay and requires
bit-exact frozen logits. It refuses to overwrite an existing output directory;
the candidate above has already been exported and validated.

## Accuracy and Reports

- `eval_single_hw_outputs.py`: evaluate low-resolution logits and full-resolution
  masks against the same sample and center-cropped ground truth.
- `eval_val_hw_masks_fullres.py`: validation-set PA, per-class accuracy/IoU and mIoU.
- `eval_hw_constrained_qat.py`: common model and label-mapping support.
- `audit_hls_reports.py`: inspect synthesis timing, resources and risk messages.
- `check_p6_hls_structure.py`: current P7 source ownership and singleton checks;
  its historical filename is retained.
- `analyze_conv_cycle_profile.py`: analyze EXEC-prefix and WIN/SA/POST counters.
- `analyze_sa_utilization.py`: calculate MAC work, filling and service lower bounds.

For evaluation, explicitly supply the current artifact, cache, output paths and
`--out-json`; do not rely on historical CLI path defaults. WIN/SA/POST counters
classify concurrent activity in the same Conv interval and must not be summed.
HLS maximum latency is not a replacement for board cycle measurements.

Independent output-mask geometry is explicit: `fullres_mask_from_logits(...,
mask_shape=(512, 512))`, or `--mask-height 512 --mask-width 512` in the evaluators.
The encoder `geometry` must stay 240x240/30x30; do not fake a 512 encoder.
Manifests may declare `mask_shape` and `target_encoding=cityscapes_train_ids`
for raw annotations. Already mapped `.npy` targets retain their separate label
semantics. High-resolution targets are read from original annotation PNGs,
center-cropped and half-pixel nearest-sampled, never enlarged from 240 labels.
The isolated 500-image study, references and reproduction commands are in
`D:/ESPNet/resolution_h240w240_gray/output_restore_1104/README.md`.

## Checks

```powershell
& 'D:/ESPNet/.venv/Scripts/python.exe' -m unittest `
  tools.test_h240w240_gray_contract tools.test_eval_single_hw_outputs
& 'D:/ESPNet/.venv/Scripts/python.exe' tools/test_p7_contracts.py
& 'D:/ESPNet/.venv/Scripts/python.exe' tools/check_p6_hls_structure.py
& 'D:/ESPNet/.venv/Scripts/python.exe' -B -m unittest tools.test_app_h240w240_gray_contract
```

Other `test_*.py` files cover shared arithmetic/profiling or preserved H256
compiler contracts. The App contract test follows this branch's H240 single-image
configuration, not the old dual-model test. They do not imply that old RGB paths remain
in the new hardware. `testdata/rtl_stage_0910.csv` is a frozen historical board
measurement, not a performance measurement for the grayscale IP.

The production `ESP_INT8_hls/hls_config.cfg` keeps the original top and source list.
Its TB now defaults to the frozen square-source study and performs one full mode-3
inference. It writes `csim_u72_lowres_logits.bin`, `hls_output_logits_s3.bin` and
`hls_output_mask_s3.bin` under the chosen CSim work directory. From `ESP_INT8_hls`:

```powershell
$env:ESP_INT8_INPUT_RESIZE_TEST = 'full'
& 'D:/Xilinx/2025.1/Vitis/bin/vitis-run.bat' --mode hls --tcl run_csim_input_resize.tcl
```

Set the test to `unit` for DMA checks or `mask240` for the old 240-mask regression
without a network run. The full TB also checks compact tails/error draining,
bilinear extrema/ties/segment boundaries, row readiness, guards and reset.

Synthesis, packaging, implementation and app build remain manual. CSim validates
functional/token contracts, not RTL FIFO liveness or measured cycle savings.
