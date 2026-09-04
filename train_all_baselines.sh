#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="/root/autodl-tmp/MESM"
GPU_ID="${1:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIGS=(
  "config/charades/VGG_GloVe-ambedy2.json"
)

NAMES=(
  "charades_VGG_GloVe_ambedy2"
)

cd "${REPO_DIR}" || {
  echo "[ERROR] Repository not found: ${REPO_DIR}"
  exit 1
}

missing=0
for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "[ERROR] Missing config: ${cfg}"
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "[ABORT] One or more training configs are missing."
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_DIR}/batch_train_logs/${STAMP}"
mkdir -p "${LOG_DIR}"

SUMMARY="${LOG_DIR}/summary.tsv"
printf "index\tname\tconfig\tstatus\texit_code\tstart_time\tend_time\tlog\n" > "${SUMMARY}"

echo "============================================================"
echo " MESM sequential baseline training"
echo " Repository : ${REPO_DIR}"
echo " GPU        : ${GPU_ID}"
echo " Python     : $(${PYTHON_BIN} --version 2>&1)"
echo " Log dir    : ${LOG_DIR}"
echo " Runs       : ${#CONFIGS[@]}"
echo "============================================================"

success=0
failed=0

for i in "${!CONFIGS[@]}"; do
  idx=$((i + 1))
  cfg="${CONFIGS[$i]}"
  name="${NAMES[$i]}"
  log="${LOG_DIR}/${idx}_${name}.log"

  start_human="$(date '+%Y-%m-%d %H:%M:%S')"
  start_epoch="$(date +%s)"

  echo
  echo "============================================================"
  echo "[${idx}/${#CONFIGS[@]}] START: ${name}"
  echo "Config : ${cfg}"
  echo "GPU    : ${GPU_ID}"
  echo "Time   : ${start_human}"
  echo "Log    : ${log}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU_ID}"   PYTHONUNBUFFERED=1   "${PYTHON_BIN}" train.py --config_file "./${cfg}"     2>&1 | tee "${log}"

  code=${PIPESTATUS[0]}
  end_human="$(date '+%Y-%m-%d %H:%M:%S')"
  end_epoch="$(date +%s)"
  elapsed=$((end_epoch - start_epoch))

  if [[ "${code}" -eq 0 ]]; then
    status="SUCCESS"
    success=$((success + 1))
  else
    status="FAILED"
    failed=$((failed + 1))
  fi

  printf "%d\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n"     "${idx}" "${name}" "${cfg}" "${status}" "${code}"     "${start_human}" "${end_human}" "${log}" >> "${SUMMARY}"

  printf "[%d/%d] %s: %s | elapsed=%02dh:%02dm:%02ds | exit=%d\n"     "${idx}" "${#CONFIGS[@]}" "${status}" "${name}"     $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60)) "${code}"

  sleep 5
done

echo
echo "============================================================"
echo " ALL RUNS FINISHED"
echo " Success : ${success}"
echo " Failed  : ${failed}"
echo " Summary : ${SUMMARY}"
echo " Logs    : ${LOG_DIR}"
echo "============================================================"

if [[ "${failed}" -ne 0 ]]; then
  exit 2
fi
