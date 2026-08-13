#!/bin/sh
# One image answers all three: SageMaker runs it as `... train` or `... serve` (Batch
# Transform), and the Processing job overrides the entrypoint to run run_monitor.py. The
# SDK's local-mode ModelTrainer runs the image with no argument, so a bare run trains too.
# Anything else is passed through (handy for a shell).
set -e
if [ "$1" = "serve" ]; then
  exec gunicorn --bind 0.0.0.0:8080 --workers "${GUNICORN_WORKERS:-2}" --timeout 120 serve:app
elif [ "$1" = "train" ] || [ -z "$1" ]; then
  exec python train.py
else
  exec "$@"
fi
