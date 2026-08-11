#!/bin/sh
# One image answers both: SageMaker runs it as `... train` or `... serve`; the SDK local-mode trainer runs it with no argument, so a bare run trains too; anything else is passed through (handy for a shell).
set -e
if [ "$1" = "serve" ]; then
  exec gunicorn --bind 0.0.0.0:8080 --workers "${GUNICORN_WORKERS:-2}" --timeout 120 serve:app
elif [ "$1" = "train" ] || [ -z "$1" ]; then
  exec python train.py
else
  exec "$@"
fi
