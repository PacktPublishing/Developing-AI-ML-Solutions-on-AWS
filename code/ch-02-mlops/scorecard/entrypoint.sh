#!/bin/sh
# SageMaker runs the training image as `... train` and the serving image as
# `... serve`. One image answers both, so the same artifact-producing container
# is the one that serves. Anything else is passed through (handy for a shell).
set -e
if [ "$1" = "serve" ]; then
  exec gunicorn --bind 0.0.0.0:8080 --workers "${GUNICORN_WORKERS:-2}" --timeout 120 serve:app
elif [ "$1" = "train" ]; then
  exec python train.py
else
  exec "$@"
fi
