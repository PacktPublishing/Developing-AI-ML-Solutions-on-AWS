# /// script
# dependencies = ["boto3", "ollama"]
# ///
"""Provide Bedrock and local Ollama clients for text, document and embedding calls."""

import io
import json
import math
import os

import boto3

# -------------------------------------------------------------------------------
# Model configuration
# -------------------------------------------------------------------------------
# Qwen3 runs on both Bedrock and Ollama; Titan pairs with 1024-dim mxbai-embed-large so the vector schema matches.
TEXT_MODEL = os.environ.get("TEXT_MODEL", "qwen.qwen3-next-80b-a3b")
# The document reader is a different job from the chat model: it is handed a PDF
# as a Converse document block and asked for the figures in it. Claude Haiku 5.5
# reads the file itself, so a supporting document needs no parser of its own.
# Global cross-Region inference, the profile the chapter's stacks allow.
DOC_MODEL = os.environ.get("DOC_MODEL", "global.anthropic.claude-haiku-5-5")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))

# qwen3:4b is the smallest tag that calls tools reliably: at 0.6b the agent
# answers that the statement is unavailable rather than calling read_case, so
# retrieval still works but the agent does not. Point at a larger tag for better
# prose; pull with `ollama pull qwen3:4b`.
OLLAMA_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "qwen3:4b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large")


# -------------------------------------------------------------------------------
# Local Bedrock shim
# -------------------------------------------------------------------------------
def _as_text(blocks: list[dict]) -> str:
    """Convert message blocks to text for Ollama, extracting text from PDF documents."""
    parts = []
    for block in blocks:
        if "text" in block:
            parts.append(block["text"])
        if "document" in block:
            doc = block["document"]
            name = doc.get("name", "document")
            parts.append(f"--- {name} ---\n{_document_text(doc)}")
    return "\n\n".join(parts)


def _document_text(doc: dict) -> str:
    """Return the text of a Converse document block, for the local path."""
    source = doc.get("source", {})
    if "text" in source:
        return source["text"]
    raw = source.get("bytes")
    if raw is None:
        return ""
    if doc.get("format") != "pdf":
        return raw.decode("utf-8", "replace")
    import pdfplumber  # lazy import: only the local document path needs it

    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


class LocalBedrockRuntime:
    """A bedrock-runtime stand-in backed by Ollama."""

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
            {"role": m["role"], "content": _as_text(m["content"])} for m in messages
        )
        cfg = inferenceConfig or {}
        options = {}
        if "maxTokens" in cfg:
            options["num_predict"] = cfg["maxTokens"]
        if "temperature" in cfg:
            options["temperature"] = cfg["temperature"]
        # think=False matters: the chapter's default tag is a thinking model, and
        # with thinking on Ollama returns the reasoning in its own field and
        # leaves message.content empty when the budget runs out first. Bedrock
        # has no such field, so the shim asks for the answer only.
        resp = self._client.chat(
            model=OLLAMA_TEXT_MODEL, messages=chat, options=options, think=False
        )
        text = resp["message"]["content"]
        tokens_in = resp.get("prompt_eval_count") or 0
        tokens_out = resp.get("eval_count") or 0
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": (
                "max_tokens" if resp.get("done_reason") == "length" else "end_turn"
            ),
            "usage": {
                "inputTokens": tokens_in,
                "outputTokens": tokens_out,
                "totalTokens": tokens_in + tokens_out,
            },
            # Ollama reports total_duration in nanoseconds
            "metrics": {"latencyMs": (resp.get("total_duration") or 0) // 1_000_000},
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
