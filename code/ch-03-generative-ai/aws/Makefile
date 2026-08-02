.PHONY: iam check-account smoke

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

# -------------------------------------------------------------------------------
# iam: provision THIS chapter's deploy identity, when you start the chapter
# Creates ch03-user from iam/deploy.json (least-privilege for ch-03 only) and trusts you to
# assume it — per chapter, not an account-wide up-front bootstrap. Re-run after
# editing the policy. Then add a [profile ch03] (role_arn + source_profile) and
# run cloud targets with AWS_PROFILE=ch03.
# -------------------------------------------------------------------------------
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch03-user --description "Deploy/operate ch-03" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch03-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch03-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch03-user ready"