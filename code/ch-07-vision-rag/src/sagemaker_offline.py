"""Neutralize the SDK calls that assume a real AWS account (the SM_OFFLINE path only).

The same idea as Chapter 4's offline module, for the mode this chapter uses. ModelBuilder
validates a SageMaker serving execution role in every mode, Mode.IN_PROCESS included, even
though nothing here reaches SageMaker: the model is hosted in this Python process. With no
account there is no STS or IAM to reach, so skip that validation the way a notebook without
iam:SimulatePrincipalPolicy does, and skip the default-bucket lookup that also calls STS.
The AWS path (aws/deploy.py) never imports this.
"""

from sagemaker.core.helper.session_helper import Session
from sagemaker.serve import model_builder


def use_local_stubs() -> None:
    """Apply the offline-mode stubs. Call before building the in-process endpoint."""

    def keep_role(provided_role=None, **_):
        """Return the role as given, skipping the STS lookup an offline run cannot make."""
        return provided_role

    def local_bucket(_self):
        """Name the staging bucket without asking S3 for a default."""
        return "local"

    model_builder.resolve_and_validate_role = keep_role
    Session.default_bucket = local_bucket
