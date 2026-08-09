# /// script
# dependencies = ["boto3", "ollama"]
# ///
"""The model seam: one Bedrock Converse interface, two backends.

get_runtime() returns a bedrock-runtime client -- real boto3, or a local shim
backed by Ollama when BEDROCK_LOCAL=1. Both answer .converse(...) in the Bedrock
Converse response shape, so the classifier and the batch job never change.
"""

import os

import boto3

TEXT_MODEL = os.environ.get(
    "TEXT_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
)
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
OLLAMA_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "qwen3:0.6b")


class LocalBedrockRuntime:
    """A bedrock-runtime stand-in backed by Ollama, exposing .converse()."""

    def __init__(self) -> None:
        """Create the Ollama client, honoring OLLAMA_HOST."""
        import ollama

        self._client = ollama.Client(
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        )

    def converse(
        self, modelId: str, messages: list, system=None, inferenceConfig=None
    ) -> dict:
        """Run a chat completion and return it in the Converse response shape."""
        chat = []
        if system:
            chat.append(
                {"role": "system", "content": " ".join(b["text"] for b in system)}
            )
        chat.extend(
            {
                "role": m["role"],
                "content": " ".join(b.get("text", "") for b in m["content"]),
            }
            for m in messages
        )
        cfg = inferenceConfig or {}
        options = {}
        if "maxTokens" in cfg:
            options["num_predict"] = cfg["maxTokens"]
        if "temperature" in cfg:
            options["temperature"] = cfg["temperature"]
        resp = self._client.chat(
            model=OLLAMA_TEXT_MODEL, messages=chat, options=options, think=False
        )
        text = resp["message"]["content"]
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn",
        }


def get_runtime():
    """Return a bedrock-runtime client: real boto3, or the local Ollama shim."""
    if os.environ.get("BEDROCK_LOCAL") == "1":
        return LocalBedrockRuntime()
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


def answer_text(response: dict) -> str:
    """Pull the assistant's text out of a Converse response."""
    return response["output"]["message"]["content"][0]["text"].strip()
