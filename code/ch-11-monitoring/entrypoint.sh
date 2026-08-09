#!/bin/sh
# One image answers all three: SageMaker runs it as `... train` or `... serve` (Batch
# Transform), and the Processing job overrides the entrypoint to run sm_monitor.py.
# Anything else is passed through (handy for a shell).
set -e
if [ "$1" = "serve" ]; then
  exec gunicorn --bind 0.0.0.0:8080 --workers "${GUNICORN_WORKERS:-2}" --timeout 120 serve:app
elif [ "$1" = "train" ]; then
  exec python train.py
else
  exec "$@"
fi
