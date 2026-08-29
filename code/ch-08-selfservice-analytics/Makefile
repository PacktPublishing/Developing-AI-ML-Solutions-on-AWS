COMPOSE := docker compose -f local/analytics-stack.yml

# awsdac renders diagrams/yaml into diagrams/png. Custom icons ship as SVG and
# are rasterised at ICON_HEIGHT (needs rsvg-convert); WIDTH sets pixel width
# (0 = awsdac native).
AWSDAC_VERSION := v0.22.4
AWSDAC := $(shell go env GOPATH)/bin/awsdac
WIDTH ?= 0
ICON_HEIGHT ?= 500

.PHONY: llm local local-bedrock local-claude all local-stop diagrams lint \
	access-plan access

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

# access: the privileges in config/access.yaml, applied to the running local stack.
# redtape plans; we apply the plan ourselves, because redtape 0.4.2 raises
# NameError on any non-dry run (cli.py:220 reads an undefined idx). click is
# pinned because 8.2 broke the typer version redtape depends on.
# PYTHONHASHSEED fixes redtape's diff, which iterates a set and so depends on
# per-process string hashing: unpinned, about 1 run in 6 re-plans a grant already
# held. Pinning makes the plan reproducible, and 0 is a clean seed for this schema.
REDTAPE := PYTHONHASHSEED=0 uv run --with redtape-py --with 'click<8.2' --no-project redtape
REDTAPE_INI := printf '[redtape.redshift]\nhost = localhost\nport = 5439\ndbname = analytics\nuser = bi_analyst\npassword = local\n' > .redtape.ini

# what would change, without changing it
access-plan:
	@$(REDTAPE_INI); trap 'rm -f .redtape.ini' EXIT; \
		$(REDTAPE) run config/access.yaml --dry --operation GRANT --operation REVOKE

# apply the plan, then re-plan: a converged warehouse prints no statements. The seed
# above removes the run-to-run variance, but which seeds are clean depends on the
# schema, so the loop still applies and re-checks rather than trusting one plan. The
# grants are idempotent. A redtape failure, by contrast, fails the target rather than
# reading as "nothing to do".
# The names in access.yaml are database.schema[.table], the form redtape reports
# current state in and compares against. It emits them into GRANT verbatim, which
# is not valid SQL, so the leading database part is stripped before applying.
access:
	@$(REDTAPE_INI); trap 'rm -f .redtape.ini .access-plan.*' EXIT; \
		$(REDTAPE) run config/access.yaml --dry --operation GRANT --operation REVOKE \
			> .access-plan.raw 2> .access-plan.err \
			|| { echo "redtape could not plan:"; tail -3 .access-plan.err; exit 1; }; \
		grep -E '^[[:space:]]*(GRANT|REVOKE)\b' .access-plan.raw \
			| sed -E 's/ (ON (SCHEMA|TABLE)) analytics\./ \1 /' > .access-plan.sql || true; \
		if [ -s .access-plan.sql ]; then \
			$(COMPOSE) exec -T redshift-local psql -h 127.0.0.1 -p 5433 -U bi_analyst \
			-d analytics -v ON_ERROR_STOP=1 -q -f - < .access-plan.sql \
			&& echo "applied $$(wc -l < .access-plan.sql | tr -d ' ') statement(s)"; \
		else echo "already converged"; fi; \
			$(REDTAPE) run config/access.yaml --dry --operation GRANT --operation REVOKE \
			> .access-plan.raw 2>/dev/null; \
			echo "re-plan:"; \
			grep -E '^[[:space:]]*(GRANT|REVOKE)\b' .access-plan.raw || echo "  no changes"

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

diagrams:
	@test -x "$(AWSDAC)" || go install github.com/awslabs/diagram-as-code/cmd/awsdac@$(AWSDAC_VERSION)
	@for s in ../../assets/diagrams/icons/*.svg; do \
		rsvg-convert -h $(ICON_HEIGHT) "$$s" -o "../../assets/diagrams/icons/$$(basename $$s .svg).png"; \
	done
	@mkdir -p diagrams/png
	@for f in diagrams/yaml/*.yaml; do \
		echo "  $$f"; \
		"$(AWSDAC)" -f "$$f" -o "diagrams/png/$$(basename $$f .yaml).png" --width $(WIDTH); \
	done