# ESP INT8 Tools

Current HLS branch: `feature/h512w512-gray-int8`, binary2 INT8, native grayscale
512x512 input, 64x64 logits, 512x512 output mask, PARAM v4. There is no input resize.
Model training and hardware-constrained QAT tools live in `D:/ESPNet`.

## Artifact Contract

Use `hw_artifacts/binary2_int8_h512w512_gray_finetuned_v4` with this HLS branch.
It contains native512-adapted weights from 40 FP32 fine-tuning epochs and the
best of 10 hardware-aligned INT8 QAT epochs. The v4 layout and schedules remain
compatible with `platform_h512_baseline`. C1 packed-row reuse is 16 words;
compact C129 rows span 516 words. `INPUTQ.BIN` and the output mask are each 262,144
bytes; low-resolution HWC INT8 logits are 8,192 bytes. The manifest records grayscale
normalization, center-square crop, checkpoint/cache hashes and tensor geometry.
Intermediate compact rows carry an explicit 32-byte row-padding flag; input and
pool tensors remain tightly packed.

The model and 500-image integer-replay results live under
`D:/ESPNet/resolution_h512w512_gray/`. Deployment golden comes from the actual
PARAM replay, not the model's floating-point affine hooks. See its README and
`report_and_workplans/workplan_1110.md` for the accuracy comparison. The obsolete
H240-direct-transfer native512 hardware artifact has been removed. QAT and
replay both score mIoU=86.9902%, with 0/4096000 differing logit values; this is
not a board or CSim validation claim for the new weights.

The tools retain offline compatibility with preserved H240 grayscale and H256 RGB
releases, but those PARAM files are **not compatible with this native512 HLS IP**.
Native512 implementation passed at 100 MHz. The App is prepared for
`platform_h512_baseline`, with `P512F.BIN` / `I512F.BIN` / `O512F.BIN` on SD.
The App build remains manual; do not use an old ELF or BOOT.BIN. See
`ESP_INT8_app/src/README.txt` and `boot_h512_native.bif` for JTAG/SD boot preparation.

## Compile and Validate

- `export_int8_hw_blob.py`: compile PARAM v4, schedules and memory/fusion audits.
- `hw_param_replay.py`: execute PARAM using HLS-compatible integer arithmetic.
- `hw_int8_math.py`: arithmetic shared by compiler and replay.
- `geometry_contract.py`: manifest-driven input channels and geometry.
- `export_exec_prefix_params.py`: generate EXEC-prefix profiling PARAM files.
- `export_val_hw_dataset.py`: export validation inputs using the QAT loader.

Run from `D:/ESP_INT8`:

```powershell
& 'D:/ESPNet/.venv/Scripts/python.exe' tools/hw_param_replay.py `
  --param hw_artifacts/binary2_int8_h512w512_gray_finetuned_v4/PARAM.BIN `
  --input hw_artifacts/binary2_int8_h512w512_gray_finetuned_v4/INPUTQ.BIN `
  --output hw_artifacts/binary2_int8_h512w512_gray_finetuned_v4/replay_check.bin
```

The model-side export driver uses the packed-PARAM accelerated replay backend;
the standalone replay CLI above is the slower scalar reference. Both decode the
same packed weights and integer qparams. Export requires a new output directory.
No training, calibration or observer updates are performed by replay. The
selected export already exists; do not rerun the frozen-H240 export workflow.

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
  tools.test_h512w512_gray_contract tools.test_h240w240_gray_contract `
  tools.test_app_h240w240_gray_contract
& 'D:/ESPNet/.venv/Scripts/python.exe' tools/check_p6_hls_structure.py
& 'D:/ESPNet/.venv/Scripts/python.exe' -B `
  D:/ESPNet/resolution_h512w512_gray/test_baseline_1110.py
```

Other `test_*.py` files cover shared arithmetic/profiling or preserved H240/H256
compiler contracts. Their names do not imply old handlers remain in the new
hardware. `testdata/rtl_stage_0910.csv` is a frozen historical board
measurement, not a performance measurement for the grayscale IP.

Full-resolution CSim is configured by `ESP_INT8_hls/hls_config.cfg` and writes
`csim_u72_lowres_logits.bin` and `hls_output_mask.bin` under the chosen CSim
work directory. Its TB defaults to the native512 artifact. To run the final
validation without changing the production cfg, use this from `ESP_INT8_hls`:

```powershell
$env:ESP_INT8_NATIVE512_TEST = 'full'
& 'D:/Xilinx/2025.1/Vitis/bin/vitis-run.bat' --mode hls --tcl run_csim_1110.tcl
```

The same Tcl accepts `dma`, `focus` and `boundary` for targeted checks. Its
workspaces are named `hls_work_h512w512_gray_baseline_*`; old H240 results are not
overwritten. Allow at least two hours for full-frame CSim. The TB also checks
516-word alias-safe Vec commit, BLOCK5 row-group reuse, malformed/old PARAM,
bilinear edge/tie/reset behavior, and exact integer-reference logits and mask.

Synthesis, packaging, implementation and app build remain manual. CSim validates
functional/token contracts, not RTL FIFO liveness or measured cycle savings.
