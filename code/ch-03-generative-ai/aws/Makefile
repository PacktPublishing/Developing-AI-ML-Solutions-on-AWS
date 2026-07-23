.PHONY: check-account smoke

# Bedrock model checks against real AWS. The full deploy of the assistant lives
# in assistant-api/ (run make deploy there).
ACCOUNT := $(shell aws sts get-caller-identity --query Account --output text)
RUN_STAMP := $(shell date +%Y-%m-%d)
EMBED_MODEL ?= amazon.titan-embed-text-v2:0
TEXT_MODEL ?= qwen.qwen3-next-80b-a3b

check-account:
	@echo "account: $(ACCOUNT)"

smoke:
	@{ \
	  echo "account: $(ACCOUNT)  date: $(RUN_STAMP)"; \
	  echo "embed model: $(EMBED_MODEL)"; \
	  aws bedrock-runtime invoke-model --region us-east-1 --model-id $(EMBED_MODEL) \
	  --body "$$(printf '{"inputText":"solar project company term loan","dimensions":1024,"normalize":true}' | base64)" \
	  --cli-binary-format base64 /tmp/ch03-embed.json >/dev/null && \
	  python3 -c "import json;d=json.load(open('/tmp/ch03-embed.json'));print('  embedding dims:',len(d['embedding']),'tokens:',d['inputTextTokenCount'])"; \
	  echo "text model: $(TEXT_MODEL)"; \
	  aws bedrock-runtime converse --region us-east-1 --model-id $(TEXT_MODEL) \
	  --messages '[{"role":"user","content":[{"text":"Reply with exactly: bedrock ok"}]}]' \
	  --inference-config '{"maxTokens":10}' --query 'output.message.content[0].text' --output text | sed 's/^/  converse: /'; \
	  } | tee /tmp/ch03-bedrock-$(RUN_STAMP).txt