# The scoring Lambda as a container image. Same model and model class as the
# BYOC endpoint, on the AWS Lambda base image. That image bundles the open-source
# Lambda Runtime Interface Emulator, so the exact container runs locally with a
# plain `docker run -p 9000:8080 ...` (POST to the RIE /invocations path) and on
# AWS Lambda unchanged. challenger_model.py and the trained model are copied into
# the build context by `make lambda-build` (both git-ignored, so the image is
# always built from a freshly trained model).
FROM public.ecr.aws/lambda/python:3.12

RUN pip install --no-cache-dir \
    "xgboost>=2.0" \
    "scikit-learn>=1.3" \
    "pandas>=2.0" \
    "numpy>=1.26"

COPY challenger_model.py handler.py ${LAMBDA_TASK_ROOT}/
COPY model/ /opt/ml/model/

CMD ["handler.lambda_handler"]
