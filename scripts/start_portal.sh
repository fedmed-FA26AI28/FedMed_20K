#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-institution}"
state="${FEDMED_STATE_DIR:-.portal}"
web_python="${FEDMED_WEB_PYTHON:-.venv-web/bin/python}"
if [[ ! -f web/dist/index.html ]]; then
  printf '%s\n' 'Build web/dist on the development laptop and copy it to this device before starting.' >&2
  exit 1
fi
if [[ ! -f "$state/settings.json" ]]; then
  : "${FEDMED_TRAINER_PYTHON:?Set FEDMED_TRAINER_PYTHON to the existing training Python executable.}"
  "$web_python" -m portal init --mode "$mode" --state-dir "$state" \
    --name "${FEDMED_NAME:-Jetson institution}" --device-type "${FEDMED_DEVICE_TYPE:-jetson_nano}" \
    --trainer-python "$FEDMED_TRAINER_PYTHON" --advertise-host "${FEDMED_SERVER_HOST:-127.0.0.1}"
fi
exec "$web_python" -m portal serve --state-dir "$state"
