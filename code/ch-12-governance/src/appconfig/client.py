"""The feature-flag seam: AWS AppConfig on AWS, a local flag store from source.

get_appconfig() returns a client whose configuration(context) yields the evaluated
feature flags for that request. On AWS it fetches the raw configuration through the
AppConfig management API -- which keeps the _variants the rule engine needs -- and
evaluates the variant rules in process, exactly as the AppConfig agent sidecar would;
locally it reads the same feature-flag document from a file. Rule evaluation, including
the FNV-1a split that buckets each request, is rule_evaluator.py, composed from source.

Env: APPCONFIG_LOCAL=1 -> local flag store; APPCONFIG_FLAGS -> the local file;
     APPCONFIG_APP / APPCONFIG_PROFILE and AWS_REGION for the AWS path.
"""

import json
import os
from pathlib import Path

from .rule_evaluator import evaluate_config


class LocalFlagStore:
    """Read the feature-flag document from a local file (the flag store)."""

    def __init__(self, path: str | Path) -> None:
        """Point at the local feature-flag JSON."""
        self._path = Path(path)

    def raw(self) -> dict:
        """Return the raw (unevaluated) feature-flag document."""
        return json.loads(self._path.read_text())


class AppConfigStore:
    """Fetch the raw configuration through the AppConfig management API."""

    def __init__(
        self, application: str, profile: str, region: str | None = None
    ) -> None:
        """Resolve against a named AppConfig application and configuration profile."""
        import boto3

        self._app, self._profile = application, profile
        self._client = boto3.client(
            "appconfig", region_name=region or os.environ.get("AWS_REGION", "us-east-1")
        )

    def _resolve(self, method: str, name: str, **kwargs) -> str:
        """Resolve an AppConfig resource name to its id."""
        for item in getattr(self._client, method)(**kwargs).get("Items", []):
            if item.get("Name") == name:
                return item["Id"]
        raise RuntimeError(f"AppConfig resource '{name}' not found")

    def raw(self) -> dict:
        """Return the latest hosted configuration version, with its _variants intact."""
        app = self._resolve("list_applications", self._app)
        profile = self._resolve(
            "list_configuration_profiles", self._profile, ApplicationId=app
        )
        versions = self._client.list_hosted_configuration_versions(
            ApplicationId=app, ConfigurationProfileId=profile
        )["Items"]
        latest = max(v["VersionNumber"] for v in versions)
        body = self._client.get_hosted_configuration_version(
            ApplicationId=app, ConfigurationProfileId=profile, VersionNumber=latest
        )["Content"].read()
        return json.loads(body)


class AppConfig:
    """A flag source plus per-request rule evaluation."""

    def __init__(self, store) -> None:
        """Wrap a flag store (local file or AppConfig)."""
        self._store = store

    def configuration(self, context: dict) -> dict:
        """Evaluate every flag's variant rules against the request context."""
        return evaluate_config(self._store.raw(), context)


def get_appconfig() -> AppConfig:
    """Return the feature-flag client: the local flag store, or real AppConfig."""
    if os.environ.get("APPCONFIG_LOCAL") == "1":
        path = os.environ.get("APPCONFIG_FLAGS", "local/flags/feature-flags.json")
        return AppConfig(LocalFlagStore(path))
    return AppConfig(
        AppConfigStore(
            application=os.environ.get("APPCONFIG_APP", "credit-governance"),
            profile=os.environ.get("APPCONFIG_PROFILE", "rollout-flags"),
        )
    )
