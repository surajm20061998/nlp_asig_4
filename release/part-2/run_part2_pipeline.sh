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

LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${LOG_ROOT}/part2_${RUN_ID}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-experiment}"
MODEL_NAME="${MODEL_NAME:-google-t5/t5-small}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
PATIENCE_EPOCHS="${PATIENCE_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
NUM_BEAMS="${NUM_BEAMS:-4}"
MAX_GENERATION_LENGTH="${MAX_GENERATION_LENGTH:-256}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-512}"
SEED="${SEED:-42}"
INCLUDE_SCHEMA_IN_INPUT="${INCLUDE_SCHEMA_IN_INPUT:-0}"
LOWERCASE_INPUTS="${LOWERCASE_INPUTS:-0}"
NORMALIZE_WHITESPACE="${NORMALIZE_WHITESPACE:-1}"
FREEZE_ENCODER="${FREEZE_ENCODER:-0}"
FREEZE_DECODER="${FREEZE_DECODER:-0}"
FREEZE_EMBEDDINGS="${FREEZE_EMBEDDINGS:-0}"

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

run_step 00_py_compile \
  "${PYTHON_BIN}" -m py_compile load_data.py t5_utils.py train_t5.py compute_stats.py utils.py evaluate.py

STATS_CMD=(
  "${PYTHON_BIN}" compute_stats.py
  --data_dir "${DATA_DIR}"
  --model_name "${MODEL_NAME}"
  --max_input_length "${MAX_INPUT_LENGTH}"
  --max_target_length "${MAX_TARGET_LENGTH}"
)

TRAIN_CMD=(
  "${PYTHON_BIN}" train_t5.py
  --finetune
  --experiment_name "${EXPERIMENT_NAME}"
  --model_name "${MODEL_NAME}"
  --data_dir "${DATA_DIR}"
  --learning_rate "${LEARNING_RATE}"
  --max_n_epochs "${MAX_EPOCHS}"
  --patience_epochs "${PATIENCE_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --test_batch_size "${TEST_BATCH_SIZE}"
  --num_beams "${NUM_BEAMS}"
  --max_generation_length "${MAX_GENERATION_LENGTH}"
  --max_input_length "${MAX_INPUT_LENGTH}"
  --max_target_length "${MAX_TARGET_LENGTH}"
  --seed "${SEED}"
)

if [[ "${INCLUDE_SCHEMA_IN_INPUT}" == "1" ]]; then
  STATS_CMD+=(--include_schema_in_input)
  TRAIN_CMD+=(--include_schema_in_input)
fi

if [[ "${LOWERCASE_INPUTS}" == "1" ]]; then
  STATS_CMD+=(--lowercase_inputs)
  TRAIN_CMD+=(--lowercase_inputs)
fi

if [[ "${NORMALIZE_WHITESPACE}" == "0" ]]; then
  STATS_CMD+=(--no_normalize_whitespace)
  TRAIN_CMD+=(--no_normalize_whitespace)
fi

if [[ "${FREEZE_ENCODER}" == "1" ]]; then
  TRAIN_CMD+=(--freeze_encoder)
fi

if [[ "${FREEZE_DECODER}" == "1" ]]; then
  TRAIN_CMD+=(--freeze_decoder)
fi

if [[ "${FREEZE_EMBEDDINGS}" == "1" ]]; then
  TRAIN_CMD+=(--freeze_embeddings)
fi

run_step 01_compute_stats "${STATS_CMD[@]}"

run_step 02_train_and_generate "${TRAIN_CMD[@]}"

run_step 03_dev_metric_check \
  "${PYTHON_BIN}" evaluate.py \
  --predicted_sql "results/t5_ft_${EXPERIMENT_NAME}_dev.sql" \
  --predicted_records "records/t5_ft_${EXPERIMENT_NAME}_dev.pkl" \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl

for file_name in \
  "results/t5_ft_${EXPERIMENT_NAME}_dev.sql" \
  "results/t5_ft_${EXPERIMENT_NAME}_test.sql" \
  "records/t5_ft_${EXPERIMENT_NAME}_dev.pkl" \
  "records/t5_ft_${EXPERIMENT_NAME}_test.pkl"
do
  if [[ -f "${SCRIPT_DIR}/${file_name}" ]]; then
    cp "${SCRIPT_DIR}/${file_name}" "${RUN_DIR}/"
  fi
done

{
  echo "Run directory: ${RUN_DIR}"
  echo
  echo "Generated artifacts:"
  ls -lh "${RUN_DIR}" 2>/dev/null || true
} | tee "${RUN_DIR}/summary.txt"
