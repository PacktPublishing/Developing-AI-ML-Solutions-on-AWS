COMPOSE := docker compose -f local/analytics-stack.yml

.PHONY: llm local local-bedrock local-claude all local-stop lint

# the one-command local flow: bring the whole stack up fully offline (Ollama-backed
# gpt-oss:20b, no account), then open http://localhost:7681 and ask a question. make
# local-stop when done. local-bedrock / local-claude swap the model backend.
all: local

# Local means local: Claude Code drives gpt-oss:20b on the host through Ollama's Anthropic-compatible API. No account, no login, no cloud call.
# Prerequisite: ollama pull gpt-oss:20b
local:
	BEDROCK_LOCAL=1 $(COMPOSE) up -d --build
	@echo "loading the model in the background; the first question skips the load wait..."
	@docker exec -d -e ANTHROPIC_BASE_URL=http://llmproxy:8091 \
		-e ANTHROPIC_AUTH_TOKEN=local -e AWS_ACCESS_KEY_ID=local \
		-e AWS_SECRET_ACCESS_KEY=local \
		-e CLAUDE_CONFIG_DIR=/home/analyst/state/warmup \
		ch08-analytics-local-assistant-1 \
		claude -p "Reply with exactly: OK" --model gpt-oss:20b \
		--dangerously-skip-permissions

# The Fargate-faithful variant: Claude runs on real Bedrock, credentials arrive through the ecs-local-endpoints sidecar as the task role supplies them on ECS. The pre-cloud parity check.
local-bedrock:
	AWS_PROFILE=$${AWS_PROFILE:-default} CLAUDE_CODE_USE_BEDROCK=1 \
		$(COMPOSE) up -d --build

# Optional: your own Claude account instead of a local model (export
# CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`, or ANTHROPIC_API_KEY).
local-claude:
	$(COMPOSE) up -d --build

local-stop:
	$(COMPOSE) down -v

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

# iam and the cloud deploy live in aws/Makefile (make -C aws iam / deploy).