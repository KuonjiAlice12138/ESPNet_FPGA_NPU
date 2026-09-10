# ESP INT8 Tools

## 当前 Round1/2 状态（2026-09-08）

本轮统一采用 `H=256, W=512, logits=32x64, scale=8`，PARAM **v4**，
75 UOP、16 EXEC，U71 与 U72 独立执行。

- 二分类可发布候选：`hw_artifacts/binary2_int8_h256w512_v4`。
- 20-class 暂存候选：`hw_artifacts/cityscapes20_int8_h256w512_v4`；模型侧 INT8 mIoU 仍未达到 Round0 的绝对 `0.42` gate，暂不作为发布 artifact。
- 模型侧输入：`round1_binary2_model_artifact`、`round1_cityscapes20_model_candidate`；两者均由 `D:/ESPNet` 的 H256/W512 模型 artifact 生成。
- 每组 `PARAM.BIN`、`INPUTQ.BIN`、golden 和 manifest 必须成套使用。manifest 记录三者 SHA256 及 geometry。
- `model_golden_output_q.*` 保留 PyTorch fake-quant hook 结果；`golden_output_q.*` 是最终 PARAM v4 整数 replay 结果。只有后者用于 HLS bit-exact 验收。
- binary2 与 20-class 的 replay、U72 HLS logits 和 H256W512 mask 均已达到 `0 mismatch`；旧 H512W1024 PARAM 会被新 HLS 几何门禁拒绝。

旧 `hw_artifacts/*_int8_0809` 仅作为兼容基线保留，不得与 H256/W512 文件混用。

## 当前数据

- 二分类 Round1：`hw_artifacts/binary2_int8_h256w512_v4`。
- 20 类 Round1 候选：`hw_artifacts/cityscapes20_int8_h256w512_v4`。
- 历史基线：`hw_artifacts/binary2_int8_0809`、`hw_artifacts/cityscapes20_int8_0809`。
- 每组 PARAM、INPUTQ 和 golden 必须配套使用；旧版目录不覆盖新尺寸产物。
- v5 产物及改动前工具保存在 `backups/rollback_u71_0907/`。
- 二分类 INT4 目录只是保留产物，不用于这次 INT8 Round1。

## 编译与参考计算

`export_int8_hw_blob.py`：从模型侧权重/scale 导出 PARAM v4，
已恢复成功提交的编译逻辑；不再支持 `--upgrade-param-v4-dir` 或 v5 导出。
重新导出前必须显式指定现存模型目录、输出目录和 geometry；不要覆盖冻结的 0809 数据。

`hw_int8_math.py`、`hw_param_replay.py`：与恢复的 HLS 对应的整数算术和 PARAM v4 replay。最终 hardware golden 必须由已写出的 `PARAM.BIN` replay 生成，不能直接沿用模型 hook 输出。
`export_p7_param_replay_prefix.py`：指定 tensor/prefix 的参考输出。

`export_exec_prefix_params.py`：从完整 v4 PARAM 生成 P00 到 P15 的性能归因前缀。
当前 app 为双模型单图模式，不需要这些前缀文件。

```powershell
python tools/export_exec_prefix_params.py --help
D:/ESPNet/.venv/Scripts/python.exe tools/hw_param_replay.py --help
```

模型侧训练/QAT 环境仍在 `D:/ESPNet`。模型侧导出和 QAT 命令必须显式使用
`--height 256 --width 512 --target-scale 8`；编译器不从 checkpoint 名称推断 geometry。
`export_val_hw_dataset.py` 用于验证集数据，不应与本次单图输入混用。

当前 Round1 的最小硬件导出命令示例：

```powershell
D:/ESPNet/.venv/Scripts/python.exe tools/export_int8_hw_blob.py `
  --artifact-dir D:/ESP_INT8/round1_binary2_model_artifact `
  --out-dir D:/ESP_INT8/hw_artifacts/binary2_int8_h256w512_v4 `
  --height 256 --width 512 --target-scale 8 `
  --strict-zp --strict-schedule
```

## 评估与报告

`eval_single_hw_outputs.py`、`eval_val_hw_masks_fullres.py`：单图/验证集全尺寸 mask 的 PA、IoU 评估。
`audit_hls_reports.py`：资源、II、端口和风险项审查。
`analyze_sa_utilization.py`、`analyze_perf_attribution.py`：工作量/性能分析。
`analyze_conv_cycle_profile.py` 保留供分析历史 conv-internal CSV；0806 平台不提供该组计数器。

`test_p7_contracts.py` 保留 v4 算术/布局/导出合同；涉及模型 QAT 目录的测试仍须显式准备数据。
`test_conv_profiling_contract.py` 中的 app 测试覆盖双模型单图、v4 拒绝 v5、独立初始化及计时边界。
该文件的 HLS/RTL conv-internal 端口检查属于暂停的 Round 1，不是回退源码的通过标准。
RR-2*/v5 专用测试和 `finalize_csim_golden.py` 不用于当前发布验收，不能据此改写冻结的 v4 golden。

当前交付范围、旧版失败记录及手动 build/上板步骤见
`report_and_workplans/workplan_0907.md` 的 Round 1/2/3。
