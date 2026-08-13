"""Neutralize the SDK calls that assume a real AWS account (the SM_OFFLINE path only).

Native local mode keeps the SDK intact: the container runs on this machine while the
control-plane calls hit your account. This module is imported only when SM_OFFLINE=1, for
running with no account at all: with no STS or IAM to reach and the image built here rather
than pulled from ECR, it skips the execution-role validation (the way a notebook without
iam:SimulatePrincipalPolicy does), the default S3 bucket lookup, and the image pull. The
cloud path (Mode.SAGEMAKER_ENDPOINT) is untouched either way.
"""

from sagemaker.core.helper.session_helper import Session
from sagemaker.serve import model_builder
from sagemaker.serve.mode import local_container_mode


def use_local_stubs() -> None:
    """Apply the local-mode stubs. Call before building or transforming."""

    def keep_role(provided_role=None, **_):
        return provided_role

    def local_bucket(_self):
        return "local"

    def use_local_image(self, image):
        # the image is built locally; connect to Docker but do not pull it from a registry
        self.client = local_container_mode._get_docker_client()
        self.client.ping()

    model_builder.resolve_and_validate_role = keep_role
    Session.default_bucket = local_bucket
    local_container_mode.LocalContainerMode._pull_image = use_local_image
