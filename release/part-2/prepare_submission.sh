#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment_name> [model_type]" >&2
  exit 1
fi

EXPERIMENT_NAME="$1"
MODEL_TYPE="${2:-ft}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SRC_SQL="results/t5_${MODEL_TYPE}_${EXPERIMENT_NAME}_test.sql"
SRC_PKL="records/t5_${MODEL_TYPE}_${EXPERIMENT_NAME}_test.pkl"

DST_SQL="results/t5_ft_experiment_test.sql"
DST_PKL="records/t5_ft_experiment_test.pkl"

if [[ ! -f "${SRC_SQL}" ]]; then
  echo "Missing source SQL file: ${SRC_SQL}" >&2
  exit 1
fi

if [[ ! -f "${SRC_PKL}" ]]; then
  echo "Missing source PKL file: ${SRC_PKL}" >&2
  exit 1
fi

cp "${SRC_SQL}" "${DST_SQL}"
cp "${SRC_PKL}" "${DST_PKL}"

echo "Prepared submission files:"
echo "  ${DST_SQL}"
echo "  ${DST_PKL}"

