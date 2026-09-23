# ESP INT8 Tools

本目录只保留当前 `H=256, W=512, logits=32x64, PARAM v4` 硬件版本所需的
编译、验证和性能分析工具。模型训练与 QAT 工具位于 `D:/ESPNet`，历史 PARAM v3/v5、
U71、INT4 和一次性调试脚本不再保留在这里。

## 硬件产物

- `export_int8_hw_blob.py`：编译 PARAM v4，并可从冻结 artifact 重编译窗口调度字段。
- `hw_param_replay.py`：按 HLS 整数语义 replay PARAM v4。
- `hw_int8_math.py`：导出与 replay 共用的 INT8 算术。
- `geometry_contract.py`：统一 H256W512 几何约束。
- `export_exec_prefix_params.py`：生成 P00-P15 EXEC prefix 性能归因文件。
- `export_val_hw_dataset.py`：导出验证集板级输入。

当前冻结输入必须成套使用：

- `hw_artifacts/binary2_int8_h256w512_r2_v4`
- `hw_artifacts/cityscapes20_int8_h256w512_r2_v4`

不得混用其他目录的 `PARAM.BIN`、`INPUTQ.BIN`、golden 或 manifest。

## 精度评估

- `eval_single_hw_outputs.py`：单图 low-resolution logits/full-resolution mask 评估。
- `eval_val_hw_masks_fullres.py`：验证集 full-resolution mask 的 PA/IoU/mIoU。
- `eval_hw_constrained_qat.py`：上述数据导出与评估工具共用的模型和标签映射辅助函数。

## 综合与性能分析

- `audit_hls_reports.py`：扫描最新 HLS 报告中的 blocker 和高风险项。
- `check_p6_hls_structure.py`：检查当前 P7 singleton、MainCtrl 和数据流结构约束；文件名因历史兼容保留。
- `analyze_conv_cycle_profile.py`：解析 EXEC-prefix 与 Conv 内部 WIN/SA/POST counter。
- `analyze_sa_utilization.py`：计算 SA 工作量、静态填充率和服务周期下界。

WIN、SA、POST counter 是同一 Conv 时间窗口内的并发状态分类，三组周期不能相加。
HLS `max latency` 也不能替代板级 RTL cycle counter。

## 回归测试

保留的 `test_*.py` 只覆盖当前仍在主路径中的功能：PARAM v4、H256W512、双模型 app、
C3 row reuse、Post16、SA reduction tree、profiling 接口、20-class 部署和评估工具。

```powershell
D:/ESPNet/.venv/Scripts/python.exe tools/test_p7_contracts.py
D:/ESPNet/.venv/Scripts/python.exe -m unittest `
  tools.test_analyze_conv_cycle_profile `
  tools.test_app_dual_profile_h256w512_contract `
  tools.test_c3_row_reuse_round2 `
  tools.test_conv_post_round1_contract `
  tools.test_conv_profiling_contract `
  tools.test_eval_single_hw_outputs `
  tools.test_multiclass_deployment_contract `
  tools.test_sa_reduction_round3 `
  tools.test_sa_utilization_round0
```

常用报告审查：

```powershell
D:/ESPNet/.venv/Scripts/python.exe tools/check_p6_hls_structure.py
D:/ESPNet/.venv/Scripts/python.exe tools/audit_hls_reports.py --strict
```

`tools/testdata/rtl_stage_0910.csv` 是当前利用率模型的冻结板级输入，不是临时输出。
