# ESP INT8 Tools

本目录只保留当前 P7/PARAM v3 工作流需要的脚本。默认基线为：

- 模型 artifact：`D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_p7_hwconv_0623`
- 硬件 artifact：`D:\ESP_INT8\hw_artifacts\sched_v3_single_p7_hwconv_0623`

## 数据与模型导出

`export_quantized_artifacts.py`

从 FP32 checkpoint 开始执行硬件约束 QAT / fake-quant eval，并导出模型侧权重、scale、golden sample。

```powershell
D:\ESPNet\.venv\Scripts\python.exe tools\export_quantized_artifacts.py `
  --output_dir D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_p7_hwconv_0623 `
  --qat --fake_quant_eval --qat_epochs 3 --calibration_batches 11
```

`export_int8_hw_blob.py`

读取模型 artifact，生成 `PARAM.BIN`、`INPUTQ.BIN`、`golden_output_q.bin` 等硬件侧文件。

```powershell
D:\ESPNet\.venv\Scripts\python.exe tools\export_int8_hw_blob.py `
  --artifact-dir D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_p7_hwconv_0623 `
  --out-dir D:\ESP_INT8\hw_artifacts\sched_v3_single_p7_hwconv_0623 `
  --param-version 3 --strict-zp --strict-schedule
```

`export_val_hw_dataset.py`

生成完整验证集上板输入/参考输出文件，供 PS app 批量读取。

## 硬件行为与对齐检查

`hw_int8_math.py`

Python 侧 HLS 等价 INT8 算术定义，包括 round、ADD bypass、Conv2d INT8 MAC/requant，以及 QAT hook。

`hw_param_replay.py`

解析 PARAM v3 并按 exec plan 做离线整数 replay，用于判断 golden 与硬件 ISA 是否一致。

`audit_p7_precision_contract.py`

一键检查当前 HLS rounding、QAT hook 覆盖，以及 PARAM replay 与 golden 的 prefix/full 对齐。

```powershell
D:\ESPNet\.venv\Scripts\python.exe tools\audit_p7_precision_contract.py `
  --stop-logical-uop 72 `
  --out-json D:\ESP_INT8\report_and_workplans\p7_precision_contract_audit_hwconv_full_0623.json
```

`export_p7_param_replay_prefix.py`

导出指定 prefix tensor 的 replay 结果，便于和 CSim dump 或 golden 分层比对。

## 精度评估

`eval_hw_constrained_qat.py`

评估低分辨率 encoder logits 的 PA/mIoU。

`eval_hw_constrained_qat_fullres.py`

评估低分辨率 logits 以及 full-resolution bilinear logits -> argmax 的 PA/mIoU。

`eval_single_hw_outputs.py`、`eval_val_hw_masks_fullres.py`

用于上板输出文件的离线 PA/mIoU 统计。

## HLS/性能辅助

`audit_hls_reports.py`

汇总 HLS 综合报告中的 II、latency、资源和 high-risk warning。

`analyze_perf_attribution.py`、`analyze_sa_utilization.py`

基于 PARAM/uop 和板端 profiling counter 做性能归因、阵列利用率分析。

`check_p6_hls_structure.py`

静态检查 HLS 源码结构，主要用于确认旧路径/dead logic 是否残留。

## 回归测试

当前必要的轻量回归测试已合并为单入口：

```powershell
python tools\test_p7_contracts.py
```

覆盖内容包括 HLS 等价 INT8 算术、PARAM v3 parser、prefix replay、Conv2d forward、prefix export smoke、precision-contract audit、WinGen schedule/weight pack 合同，以及 fullres 输出评估输入合同。旧的 HLS CSim dump 单算子测试已删除；当前以 PARAM replay vs hardware-QAT golden 作为默认对齐检查。
