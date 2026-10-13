#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
DATA_URL="https://physionet.org/files/mitdb/1.0.0/"

python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

if ! command -v wget >/dev/null 2>&1; then
    echo "Erreur : wget doit être installé pour télécharger MIT-BIH." >&2
    exit 1
fi

mkdir -p "${DATA_DIR}"

wget \
    --recursive \
    --timestamping \
    --continue \
    --no-parent \
    --no-host-directories \
    --cut-dirs=3 \
    --directory-prefix="${DATA_DIR}" \
    "${DATA_URL}"

python "${SCRIPT_DIR}/scripts/download_data.py" --decode-only
