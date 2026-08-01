#!/bin/sh
# SageMaker starts the serving image as `... serve` — in local mode and behind
# the real endpoint alike. Anything else is passed through (handy for a shell).
set -e
if [ "$1" = "serve" ] || [ -z "$1" ]; then
  exec gunicorn --bind 0.0.0.0:8080 --workers "${GUNICORN_WORKERS:-2}" --timeout 60 serve:app
else
  exec "$@"
fi
