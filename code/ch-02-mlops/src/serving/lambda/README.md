# Lambda serving

The scoring model as a Lambda container image: the same model and model class
as the BYOC endpoint, on the AWS Lambda Python base image.

- `Dockerfile`: builds on the Lambda base image, whose bundled Runtime
  Interface Emulator serves the local test
- `handler.py`: loads the model once per container and scores the request body

`make lambda-build` copies the model and model class into the build context
(both git-ignored); `make lambda-local` runs the exact production container on
port 9010 through the emulator. On AWS the image goes to ECR and the function
is created from it (`aws/README.md` covers the IAM).
