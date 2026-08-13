"""The feature-flag client: the AWS AppConfig agent evaluates every flag.

get_appconfig() returns a client whose configuration(context) yields the evaluated
feature flags for a request. The same published agent image runs as a sidecar locally
(local development mode, reading the Ion file etl/to_ion.py renders) and on ECS
(polling the deployed configuration), so the split buckets every loan identically in
both worlds and no evaluation logic lives in this codebase.

Env: APPCONFIG_AGENT_URL (default http://localhost:2772), plus APPCONFIG_APP /
     APPCONFIG_ENV / APPCONFIG_PROFILE naming the configuration.
"""

import json
import os
import urllib.request


class AgentAppConfig:
    """Query the AWS AppConfig agent, which evaluates the flag rules."""

    def __init__(
        self, base_url: str, application: str, environment: str, profile: str
    ) -> None:
        """Build the agent's configuration URL for one application/environment/profile."""
        self._url = (
            f"{base_url.rstrip('/')}/applications/{application}"
            f"/environments/{environment}/configurations/{profile}"
        )

    def configuration(self, context: dict) -> dict:
        """Ask the agent for the evaluated flags, passing the context it buckets on."""
        header = "&".join(f"{k}={v}" for k, v in context.items())
        req = urllib.request.Request(self._url, headers={"Context": header})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())


def get_appconfig() -> AgentAppConfig:
    """Return the agent-backed feature-flag client."""
    return AgentAppConfig(
        os.environ.get("APPCONFIG_AGENT_URL", "http://localhost:2772"),
        os.environ.get("APPCONFIG_APP", "credit-governance"),
        os.environ.get("APPCONFIG_ENV", "local"),
        os.environ.get("APPCONFIG_PROFILE", "rollout"),
    )
