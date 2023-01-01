#!/bin/bash
set -euo pipefail

if [ $# -eq 1 ]; then
  DIR=$1
elif [ $# -eq 0 ]; then
  DIR=$PWD
else
  echo "Usage: $0 [DIR]"
  exit 1
fi

poetry run autoflake -ir "$DIR"
poetry run isort "$DIR"
poetry run black -C "$DIR"
poetry run flake8 "$DIR"