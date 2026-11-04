#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}
TARGET_DIR=${FSC_TARGET_DIR:-$PROJECT_DIR/data/public_slu/fluent_speech_commands}
ARCHIVE_URL=${FSC_ARCHIVE_URL:-https://zenodo.org/api/records/11106540/files/fluentai.zip/content}
ARCHIVE_PATH=${FSC_ARCHIVE_PATH:-$TARGET_DIR/fluentai.zip}
EXTRACT_DIR=${FSC_EXTRACT_DIR:-$TARGET_DIR/extracted}

mkdir -p "$TARGET_DIR" "$EXTRACT_DIR"
cd "$PROJECT_DIR"

if [[ ! -s "$ARCHIVE_PATH" ]]; then
  tmp_archive="$ARCHIVE_PATH.tmp"
  curl -L --fail --retry 3 --continue-at - --output "$tmp_archive" "$ARCHIVE_URL"
  mv "$tmp_archive" "$ARCHIVE_PATH"
fi

if [[ ! -f "$EXTRACT_DIR/.pulse_fsc_unzipped" ]]; then
  unzip -q -n "$ARCHIVE_PATH" -d "$EXTRACT_DIR"
  touch "$EXTRACT_DIR/.pulse_fsc_unzipped"
fi

for csv_name in train_data.csv valid_data.csv test_data.csv speaker_demographics.csv; do
  found_csv=$(find "$EXTRACT_DIR" -type f -name "$csv_name" | head -n 1 || true)
  if [[ -z "$found_csv" ]]; then
    echo "Missing expected FSC metadata file after extraction: $csv_name" >&2
    exit 3
  fi
  cp "$found_csv" "$TARGET_DIR/$csv_name"
done

if [[ ! -e "$TARGET_DIR/wavs" ]]; then
  found_wavs=$(find "$EXTRACT_DIR" -type d -name wavs | head -n 1 || true)
  if [[ -n "$found_wavs" ]]; then
    ln -s "$found_wavs" "$TARGET_DIR/wavs"
  fi
fi

for required in train_data.csv valid_data.csv test_data.csv wavs; do
  if [[ ! -e "$TARGET_DIR/$required" ]]; then
    echo "FSC materialization incomplete: $TARGET_DIR/$required is missing" >&2
    exit 4
  fi
done

printf 'FSC materialized under %s\n' "$TARGET_DIR"
