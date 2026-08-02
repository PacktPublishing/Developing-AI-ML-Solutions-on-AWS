#!/bin/bash
# SageMaker starts an inference container as `docker run <image> serve`. Honor
# that (and any other command) so the same image works in SageMaker local mode
# and on a real endpoint.
if [ "$1" = "serve" ]; then
  exec uvicorn serve:app --host 0.0.0.0 --port 8080
fi
exec "$@"
