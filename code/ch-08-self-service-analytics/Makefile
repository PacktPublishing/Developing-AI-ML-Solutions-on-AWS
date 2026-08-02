COMPOSE := docker compose -f local/analytics-stack.yml

.PHONY: iam llm local local-stop lint

# Local means local: Claude Code drives gpt-oss:20b on the host through
# Ollama's Anthropic-compatible API (the model Ollama recommends for Claude
# Code; 16 GB machine). No account, no login, no cloud call; the terminal
# starts straight at the prompt, exactly like the Bedrock task does.
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

# The Fargate-faithful variant: Claude runs on real Bedrock, and credentials
# arrive through the ecs-local-endpoints sidecar exactly as the task role
# will supply them on ECS. Run before the cloud round as the parity check.
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

# --- iam: provision THIS chapter's deploy identity, when you start the chapter -
# Creates ch08-user from aws/iam/deploy.json (least-privilege for ch-08 only) and trusts you to
# assume it — per chapter, not an account-wide up-front bootstrap. Re-run after
# editing the policy. Then add a [profile ch08] (role_arn + source_profile) and
# run cloud targets with AWS_PROFILE=ch08.
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch08-user --description "Deploy/operate ch-08" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch08-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch08-user --policy-name deploy --policy-document file://aws/iam/deploy.json
	@echo "ch08-user ready"