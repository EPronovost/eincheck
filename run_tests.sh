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

poetry run mypy "$DIR"
poetry run pytest --cov=eincheck "$DIR"
poetry run python run_doctest.py