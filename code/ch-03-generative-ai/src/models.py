# /// script
# dependencies = ["boto3", "ollama"]
# ///
"""The model seam: one Bedrock-shaped interface, two backends.

Every model call in this chapter goes through a bedrock-runtime client. On AWS
that is the real boto3 client. Locally, set BEDROCK_LOCAL=1 and the same calls
run against Ollama through LocalBedrockRuntime, a small class that speaks the
Bedrock Converse and InvokeModel shapes and translates them to Ollama. The rest
of the chapter (embedding, retrieval, the agent) never branches on which backend
is running, so you can develop offline and switch to Bedrock by unsetting one
variable.

The shim is a few dozen lines the book owns. There is no external local-AWS
runtime to install, so the only local prerequisites are Ollama and its models.
"""

import io
import json
import math
import os

import boto3

# -------------------------------------------------------------------------------
# Model configuration
# -------------------------------------------------------------------------------
# Qwen3 runs on both Bedrock and Ollama, so the text model is the same family in
# either world. Embeddings have no shared model, so Titan pairs with 1024-dim
# mxbai-embed-large to keep the vector schema identical.
TEXT_MODEL = os.environ.get("TEXT_MODEL", "qwen.qwen3-next-80b-a3b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))

# qwen3:0.6b is the smallest Qwen3, a few hundred megabytes, so the offline path
# stays cheap to set up. Answers are correspondingly shorter and blunter than the
# 80B model on Bedrock; point OLLAMA_TEXT_MODEL at a larger qwen3 tag for better
# local output.
OLLAMA_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "qwen3:0.6b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large")


# -------------------------------------------------------------------------------
# Local Bedrock shim
# -------------------------------------------------------------------------------
class LocalBedrockRuntime:
    """A bedrock-runtime stand-in backed by Ollama.

    Implements the two operations this chapter uses, converse and invoke_model,
    with the request and response shapes boto3 returns, so calling code cannot
    tell the difference.
    """

    def __init__(self) -> None:
        """Create the Ollama client, honoring OLLAMA_HOST."""
        import ollama  # lazy import: only the local path needs it

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
            model=OLLAMA_TEXT_MODEL, messages=chat, options=options
        )
        text = resp["message"]["content"]
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn",
        }

    def invoke_model(self, modelId: str, body: str) -> dict:
        """Return an embedding in the InvokeModel response shape."""
        prompt = json.loads(body)["inputText"]
        vector = self._client.embeddings(model=OLLAMA_EMBED_MODEL, prompt=prompt)[
            "embedding"
        ]
        # Titan returns unit-normalized vectors; normalize here too so local and
        # cloud embeddings are directly comparable
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        vector = [x / norm for x in vector]
        payload = json.dumps({"embedding": vector}).encode()
        return {
            "body": io.BytesIO(payload)
        }  # BytesIO.read() mirrors the boto3 StreamingBody


# -------------------------------------------------------------------------------
# Runtime selection
# -------------------------------------------------------------------------------
def get_runtime():
    """Return a bedrock-runtime client: real boto3, or the local shim."""
    if os.environ.get("BEDROCK_LOCAL") == "1":
        return LocalBedrockRuntime()
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


# -------------------------------------------------------------------------------
# Embedding and generation
# -------------------------------------------------------------------------------
def embed(runtime, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, one vector per text."""
    vectors = []
    for text in texts:
        body = json.dumps(
            {"inputText": text, "dimensions": EMBED_DIM, "normalize": True}
        )
        resp = runtime.invoke_model(modelId=EMBED_MODEL, body=body)
        vectors.append(json.loads(resp["body"].read())["embedding"])
    return vectors


def generate(runtime, system: str, user: str, max_tokens: int = 600) -> str:
    """Generate a single grounded response for a system and user prompt."""
    resp = runtime.converse(
        modelId=TEXT_MODEL,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )
    return resp["output"]["message"]["content"][0]["text"]
