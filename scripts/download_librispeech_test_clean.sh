#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}
DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data/librispeech}
PYTHON=${PYTHON:-python3}
URL=${URL:-https://www.openslr.org/resources/12/test-clean.tar.gz}
ARCHIVE="$DATA_DIR/test-clean.tar.gz"
ROOT="$DATA_DIR/LibriSpeech/test-clean"
RECORDS=${RECORDS:-$PROJECT_DIR/data/librispeech_test_clean.jsonl}

mkdir -p "$DATA_DIR" "$PROJECT_DIR/data"
if [[ ! -d "$ROOT" ]]; then
  if [[ ! -s "$ARCHIVE" ]]; then
    curl -L --retry 5 --retry-delay 20 -C - -o "$ARCHIVE" "$URL"
  fi
  tar -xzf "$ARCHIVE" -C "$DATA_DIR"
fi

cd "$PROJECT_DIR"
"$PYTHON" -m pulse.build_librispeech_records --root "$ROOT" --output "$RECORDS"
