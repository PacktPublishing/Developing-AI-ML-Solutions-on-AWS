# Serving the container

`local_endpoint.py` deploys the chapter's bring-your-own-container through the
SageMaker Python SDK in `Mode.LOCAL_CONTAINER`, and `sagemaker_offline.py` is the
same code with the mode swapped for the cloud. `local_transform.py` runs a Batch
Transform job the same way. `fargate/`, `lambda/` and `lambda-registry/` are the
non-SageMaker serving options the chapter compares against.

`make serve` is not this path. It is a plain `docker run` of the same image, which
exercises the container without the SDK in the way; `make sm-local` is the SDK
path below.

## Notes that cost time to learn (SDK v3)

The same shape as `src/pipeline/README.md`, for the serving half. Every one of
these cost a debugging session, and ch-06's `aws/deploy.py` is a copy of the
recipe rather than a second derivation of it.

- **`ModelServer.MMS` is the one that fits a BYOC image.** There is no BYOC or
  CUSTOM value: the enum is TORCHSERVE, MMS, TENSORFLOW_SERVING,
  DJL_SERVING, TRITON, TGI, TEI, SMD, VLLM, SGLANG, VLLM_OMNI, LLAMACPP. `local_container_mode.create_server`
  dispatches only on `model_server`, so an unlisted value leaves `self.container`
  as `None` and the next line raises `AttributeError`. MMS is the generic runner:
  its `_start_serving` is `docker run <image> serve` on 8080, which is exactly
  the `/ping` plus `/invocations` contract the image already implements.
- **MMS mounts `<model_path>/code` at `/opt/ml/model`, not `<model_path>`.** The
  artifacts go in a `code/` subdirectory of an absolute temporary path. Pointing
  `model_path` straight at `artifacts/` leaves `/opt/ml/model` empty and the
  container fails to find the model file; a relative `model_path` is read as a
  Docker volume name.
- **`SchemaBuilder` is required, and its `sample_input` must be complete.** The
  local health check does not GET `/ping`. It invokes the endpoint with
  `schema_builder.sample_input` and counts the endpoint healthy only once the
  response decodes as JSON. A missing or partial sample makes every probe 500,
  and the deploy prints "Pinging local endpoint..." forever. Build the sample
  from the feature list the model was trained with, so every column is present.
- **`deploy_local()`, not `deploy()`.** `build()` then
  `deploy_local(wait=True, container_timeout_in_seconds=1200)` returns an
  endpoint whose `invoke(body=..., content_type="application/json")` answers.
- **Local mode still needs real credentials and a real execution role.**
  `build()` calls STS `GetCallerIdentity` twice, to validate the role and to name
  the default bucket, even though the model is served from `file://`. Dummy
  credentials fail with `InvalidClientTokenId`. Only the compute is local.
- **`_pull_image` always pulls.** An image that exists only in the local Docker
  daemon raises "Could not find image ... in repository"; pass the ECR URI.
- **Teardown is unreliable.** After the invoke returns, `endpoint.delete()` can
  leave the container running and a non-daemon thread blocked, so the script
  never exits. ch-06 flushes stdout, removes the container by ancestor image, and
  calls `os._exit(0)`. There is also a harmless `docker compose ... exited code 1`
  logged from a streamer thread after a successful invoke, which is not the hang.
