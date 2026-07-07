@echo off
setlocal EnableExtensions

set "ESPNET_DIR=D:\ESPNet"
set "ESPINT8_DIR=D:\ESP_INT8"
set "PYTHON=%ESPNET_DIR%\.venv\Scripts\python.exe"

set "ARTIFACT_DIR=%ESPINT8_DIR%\quantized_artifacts_hw_constrained_qat_p7_hwconv_0623"
set "HW_DIR=%ESPINT8_DIR%\hw_artifacts\sched_v4_p7_0702"

set "QAT_EPOCHS=3"
set "QAT_LR=1e-4"
set "BATCH_SIZE=4"
set "NUM_WORKERS=0"
set "CALIBRATION_BATCHES=11"

set "RUN_EVAL=1"
set "RUN_BLOB_TB=1"
set "CLEAN_OUTPUT=0"

if not exist "%PYTHON%" (
  echo [ERROR] Python venv not found: "%PYTHON%"
  exit /b 1
)

if "%CLEAN_OUTPUT%"=="1" (
  echo [INFO] Cleaning previous output directories...
  if exist "%ARTIFACT_DIR%" rmdir /s /q "%ARTIFACT_DIR%"
  if exist "%HW_DIR%" rmdir /s /q "%HW_DIR%"
)

set "PYTHONDONTWRITEBYTECODE=1"

echo [1/4] Running hardware-constrained QAT export...
"%PYTHON%" "%ESPINT8_DIR%\tools\export_quantized_artifacts.py" ^
  --output_dir "%ARTIFACT_DIR%" ^
  --qat ^
  --fake_quant_eval ^
  --qat_epochs %QAT_EPOCHS% ^
  --qat_lr %QAT_LR% ^
  --batch_size %BATCH_SIZE% ^
  --num_workers %NUM_WORKERS% ^
  --calibration_batches %CALIBRATION_BATCHES%
if errorlevel 1 goto fail

echo [2/4] Exporting hardware blob and single-sample bit-compare data...
"%PYTHON%" "%ESPINT8_DIR%\tools\export_int8_hw_blob.py" ^
  --artifact-dir "%ARTIFACT_DIR%" ^
  --out-dir "%HW_DIR%" ^
  --param-version 4 ^
  --strict-zp ^
  --strict-schedule
if errorlevel 1 goto fail

if "%RUN_EVAL%"=="1" (
  echo [3/4] Evaluating INT8 software baseline on full val set...
  "%PYTHON%" "%ESPINT8_DIR%\tools\eval_hw_constrained_qat.py" ^
    --artifact-dir "%ARTIFACT_DIR%" ^
    --out-json "%HW_DIR%\int8_baseline_metrics.json" ^
    --batch-size %BATCH_SIZE% ^
    --num-workers %NUM_WORKERS% ^
    --progress-every 25
  if errorlevel 1 goto fail
) else (
  echo [3/4] Skipping INT8 software baseline evaluation.
)

if "%RUN_BLOB_TB%"=="1" (
  echo [4/4] Validating param_blob with blob_file_tb if available...
  if exist "%ESPINT8_DIR%\ESP_INT8_hls\hls_work\blob_file_tb.exe" (
    "%ESPINT8_DIR%\ESP_INT8_hls\hls_work\blob_file_tb.exe" "%HW_DIR%\param_blob.bin"
    if errorlevel 1 goto fail
  ) else (
    echo [WARN] blob_file_tb.exe not found; skip HLS control-domain blob parse check.
  )
) else (
  echo [4/4] Skipping blob_file_tb validation.
)

echo [DONE] Hardware-constrained INT8 export flow completed.
echo [DONE] Model artifacts: "%ARTIFACT_DIR%"
echo [DONE] Hardware artifacts: "%HW_DIR%"
exit /b 0

:fail
echo [ERROR] Flow failed. Check the log above.
exit /b 1
