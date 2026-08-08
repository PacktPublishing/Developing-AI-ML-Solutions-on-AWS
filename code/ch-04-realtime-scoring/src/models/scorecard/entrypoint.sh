#!/bin/bash
# SageMaker starts an inference container as `docker run <image> serve`; honor that (and any other command).
if [ "$1" = "serve" ]; then
  exec uvicorn serve:app --host 0.0.0.0 --port 8080
fi
exec "$@"
