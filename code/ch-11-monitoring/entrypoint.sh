#!/bin/sh
# SageMaker runs the image as `... train`; anything else is passed through (handy for a shell).
set -e
if [ "$1" = "train" ]; then
  exec python train.py
else
  exec "$@"
fi
