.PHONY: llm local local-stop lint

# Local means local: Claude Code drives gpt-oss:20b on the host through
# Ollama's Anthropic-compatible API (the model Ollama recommends for Claude
# Code; 16 GB machine). No account, no login, no cloud call; the terminal
# starts straight at the prompt, exactly like the Bedrock task does.
# Prerequisite: ollama pull gpt-oss:20b
local:
	cd local && BEDROCK_LOCAL=1 docker compose up -d --build
	@echo "loading the model in the background; the first question skips the load wait..."
	@docker exec -d -e ANTHROPIC_BASE_URL=http://llmproxy:8091 \
		-e ANTHROPIC_AUTH_TOKEN=local -e AWS_ACCESS_KEY_ID=local \
		-e AWS_SECRET_ACCESS_KEY=local \
		-e CLAUDE_CONFIG_DIR=/home/analyst/state/warmup \
		ch08-analytics-local-assistant-1 \
		claude -p "Reply with exactly: OK" --model gpt-oss:20b \
		--dangerously-skip-permissions

# The Fargate-faithful variant: Claude runs on real Bedrock, and credentials
# arrive through the ecs-local-endpoints sidecar exactly as the task role
# will supply them on ECS. Run before the cloud round as the parity check.
local-bedrock:
	cd local && AWS_PROFILE=$${AWS_PROFILE:-default} CLAUDE_CODE_USE_BEDROCK=1 \
		docker compose up -d --build

# Optional: your own Claude account instead of a local model (export
# CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`, or ANTHROPIC_API_KEY).
local-claude:
	cd local && docker compose up -d --build

local-stop:
	cd local && docker compose down -v

lint:
	uvx ruff format skills/redshift-data-api/*.py local/dataapi/*.py
	uvx ruff check skills/redshift-data-api/*.py local/dataapi/*.py
llm:
	. ~/.venv-vllm-metal/bin/activate && vllm serve openai/gpt-oss-20b \
		--served-model-name gpt-oss-20b --enable-auto-tool-choice \
		--tool-call-parser openai --max-model-len 32768 --port 8090 \
		--enable-prefix-caching \
		--default-chat-template-kwargs '{"reasoning_effort": "low"}' \
		--speculative-config '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":4}'