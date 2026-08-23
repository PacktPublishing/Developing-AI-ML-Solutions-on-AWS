#!/bin/sh
# SageMaker starts the serving image as `... serve`, behind the real endpoint and in
# local mode alike. Anything else is passed through (handy for a shell).
set -e
if [ "$1" = "serve" ] || [ -z "$1" ]; then
  exec gunicorn --bind 0.0.0.0:8080 --workers "${GUNICORN_WORKERS:-1}" \
    --threads "${GUNICORN_THREADS:-4}" --timeout 300 serve:app
else
  exec "$@"
fi
