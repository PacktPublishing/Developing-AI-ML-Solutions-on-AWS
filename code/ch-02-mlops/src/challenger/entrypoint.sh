#!/bin/sh
# SageMaker runs the training image as `... train` and the serving image as
# `... serve`. One image answers both, so the container that produced the model
# is the one that serves it. Anything else is passed through.
set -e
if [ "$1" = "serve" ]; then
  exec gunicorn --bind 0.0.0.0:8080 --workers "${GUNICORN_WORKERS:-2}" --timeout 120 serve:app
elif [ "$1" = "train" ]; then
  exec python train.py
else
  exec "$@"
fi
