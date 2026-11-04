#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PROJECT_DIR:-}" ]]; then
  printf '%s\n' "$PROJECT_DIR"
  exit 0
fi

SCRIPT_SOURCE=${BASH_SOURCE[0]}
while [[ -h "$SCRIPT_SOURCE" ]]; do
  SCRIPT_DIR=$(cd -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd)
  SCRIPT_SOURCE=$(readlink "$SCRIPT_SOURCE")
  [[ $SCRIPT_SOURCE != /* ]] && SCRIPT_SOURCE=$SCRIPT_DIR/$SCRIPT_SOURCE
done

SCRIPT_DIR=$(cd -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

if [[ -f "$REPO_ROOT/pyproject.toml" && -d "$REPO_ROOT/src" ]]; then
  printf '%s\n' "$REPO_ROOT"
  exit 0
fi

printf '%s\n' "$PWD"
