#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

IMDB_DIR="${IMDB_DIR:-${SCRIPT_DIR}/data/aclImdb}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${LOG_ROOT}/part1_${RUN_ID}"

mkdir -p "${RUN_DIR}"

run_step() {
  local step_name="$1"
  shift

  echo "============================================================"
  echo "Running ${step_name}"
  echo "Command: $*"
  echo "Log: ${RUN_DIR}/${step_name}.log"
  echo "============================================================"

  "$@" 2>&1 | tee "${RUN_DIR}/${step_name}.log"
}

if [[ ! -d "${IMDB_DIR}/train" || ! -d "${IMDB_DIR}/test" ]]; then
  echo "Expected local IMDB dataset at ${IMDB_DIR} with train/ and test/ subdirectories." >&2
  exit 1
fi

run_step 00_py_compile \
  "${PYTHON_BIN}" -m py_compile main.py utils.py

run_step 01_debug_transformation \
  "${PYTHON_BIN}" main.py --imdb_dir "${IMDB_DIR}" --debug transformation

run_step 02_q1_train_eval \
  "${PYTHON_BIN}" main.py --imdb_dir "${IMDB_DIR}" --train --eval

run_step 03_q2_eval_transformed \
  "${PYTHON_BIN}" main.py --imdb_dir "${IMDB_DIR}" --eval_transformed --model_dir ./out

run_step 04_q3_train_augmented_eval_original \
  "${PYTHON_BIN}" main.py --imdb_dir "${IMDB_DIR}" --train_augmented --eval

run_step 05_q3_eval_augmented_transformed \
  "${PYTHON_BIN}" main.py --imdb_dir "${IMDB_DIR}" --eval_transformed --model_dir ./out_augmented

for file_name in \
  out_original.txt \
  out_transformed.txt \
  out_augmented_original.txt \
  out_augmented_transformed.txt
do
  if [[ -f "${SCRIPT_DIR}/${file_name}" ]]; then
    cp "${SCRIPT_DIR}/${file_name}" "${RUN_DIR}/"
  fi
done

{
  echo "Run directory: ${RUN_DIR}"
  echo
  echo "Generated output files:"
  ls -lh "${RUN_DIR}"/*.txt 2>/dev/null || true
  echo
  echo "Generated log files:"
  ls -lh "${RUN_DIR}"/*.log
} | tee "${RUN_DIR}/summary.txt"

