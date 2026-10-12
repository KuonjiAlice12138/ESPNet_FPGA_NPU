# ESP INT8 Tools

Current HLS branch: `feature/h240w240-gray-int8`, binary2 INT8, grayscale
240x240 input, 30x30 logits, 240x240 output mask, PARAM v4.
Model training and hardware-constrained QAT tools live in `D:/ESPNet`.

## Artifact Contract

Use all files from `hw_artifacts/binary2_int8_h240w240_gray_v4` together.
`INPUTQ.BIN` and the full-resolution mask are each 57,600 bytes;
low-resolution HWC INT8 logits are 1,800 bytes. The manifest records grayscale
normalization, center-square crop, checkpoint/cache hashes and tensor geometry.
Intermediate compact rows carry an explicit 32-byte row-padding flag; input and
pool tensors remain tightly packed.

The Python tools can still inspect the preserved H256W512 binary2/cityscapes20
releases. Those artifacts are **not compatible with the current H240/C1 HLS IP**,
despite both using PARAM v4. The App now selects `platform_h240_1012` and runs
the H240 grayscale single image using `P2G.BIN` / `I2G.BIN` / `O2G.BIN`.
Rebuild it manually before use. SD boot additionally requires a newly packaged
`BOOT.BIN`; `ESP_INT8_app/src/boot_h240.bif` fixes the platform/bitstream/ELF paths.

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
  --artifact-dir D:/ESPNet/resolution_h240w240_gray/quantized_artifacts_binary2_int8_hw `
  --out-dir hw_artifacts/binary2_int8_h240w240_gray_v4 `
  --param-version 4 --height 240 --width 240 --target-scale 8 `
  --strict-zp --strict-schedule

& 'D:/ESPNet/.venv/Scripts/python.exe' tools/hw_param_replay.py `
  --param hw_artifacts/binary2_int8_h240w240_gray_v4/PARAM.BIN `
  --input hw_artifacts/binary2_int8_h240w240_gray_v4/INPUTQ.BIN `
  --output hw_artifacts/binary2_int8_h240w240_gray_v4/replay_check.bin
```

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

Full-resolution CSim is configured by `ESP_INT8_hls/hls_config.cfg` and writes
`csim_u72_lowres_logits.bin` and `hls_output_mask.bin` under the chosen CSim
work directory. Synthesis, packaging, implementation and app build remain manual.
