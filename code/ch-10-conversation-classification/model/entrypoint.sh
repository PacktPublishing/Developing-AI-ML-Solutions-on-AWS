#!/bin/sh
# SageMaker starts the serving image as `... serve`, behind the real endpoint and
# in local mode alike. Anything else is passed through (handy for a shell).
set -e
if [ "$1" = "serve" ] || [ -z "$1" ]; then
  exec python serve.py
else
  exec "$@"
fi
