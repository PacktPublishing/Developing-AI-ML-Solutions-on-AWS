.PHONY: iam secret deploy gen seed url ask cases teardown

REGION := us-east-1
STACK := ch09-underwriting-kb
SECRET := ch09-os-master
MASTER := kbadmin

output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(STACK) --query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)

MYIP = $(shell curl -s https://checkip.amazonaws.com)/32
ENDPOINT = $(call output,DomainEndpoint)

Q ?= How is DTI assessed for a grocery business?
DEAL ?= A logistics firm seeks a 500,000 US dollar working-capital loan for six months.

# The store env that points the shared code at the domain instead of localhost.
# The master password is read from the secret inside each recipe (below), never
# expanded by make, so its punctuation cannot break word-splitting.
OSCONN = STORE=opensearch OPENSEARCH_HOST=$(ENDPOINT) OPENSEARCH_PORT=443 OPENSEARCH_SSL=1 \
	OPENSEARCH_USER=$(MASTER) BEDROCK_REGION=$(REGION) PYTHONPATH=src
GETPW = aws secretsmanager get-secret-value --secret-id $(SECRET) --region $(REGION) --query SecretString --output text

# -------------------------------------------------------------------------------
# deploy: secret -> domain (fine-grained access control, ~15-20 min to create)
# -------------------------------------------------------------------------------
# secret: the master password the template resolves into the domain. Random
# alphanumerics plus a fixed aA1! so it meets the master-password policy (upper,
# lower, digit, special) while staying safe to pass through a shell.
secret:
	@aws secretsmanager describe-secret --secret-id $(SECRET) --region $(REGION) >/dev/null 2>&1 || \
		aws secretsmanager create-secret --name $(SECRET) --region $(REGION) --secret-string \
		"$$(aws secretsmanager get-random-password --region $(REGION) --exclude-punctuation \
		--password-length 20 --query RandomPassword --output text)aA1!" >/dev/null
	@echo "secret $(SECRET) ready"

deploy: secret
	aws cloudformation deploy --region $(REGION) --stack-name $(STACK) \
		--template-file template.yaml --no-fail-on-empty-changeset \
		--parameter-overrides AllowedCidr=$(MYIP) MasterUserName=$(MASTER)

# gen + seed: build the synthetic corpus, then embed it into the domain with the
# same code that runs locally (embeddings come from Bedrock, indexing from home).
gen:
	cd .. && uv run etl/gen_memos.py --out data/generated/memos --count 200 --seed 7 --messy

seed:
	cd .. && PW=$$($(GETPW)); $(OSCONN) OPENSEARCH_PASSWORD="$$PW" \
		uv run etl/embed_memos.py --memos data/generated/memos

url:
	@echo "Dashboards: $(call output,DashboardsURL)"
	@echo "sign in as $(MASTER); password is in secret $(SECRET)"

ask:
	cd .. && PW=$$($(GETPW)); $(OSCONN) OPENSEARCH_PASSWORD="$$PW" \
		uv run src/retrieve.py ask --query "$(Q)"

cases:
	cd .. && PW=$$($(GETPW)); $(OSCONN) OPENSEARCH_PASSWORD="$$PW" \
		uv run src/retrieve.py cases --deal "$(DEAL)"

teardown:
	aws cloudformation delete-stack --region $(REGION) --stack-name $(STACK)
	aws cloudformation wait stack-delete-complete --region $(REGION) --stack-name $(STACK)
	-aws secretsmanager delete-secret --secret-id $(SECRET) --region $(REGION) --force-delete-without-recovery
	@echo "$(STACK) torn down"

# -------------------------------------------------------------------------------
# iam: provision THIS chapter's deploy identity, when you start the chapter
# -------------------------------------------------------------------------------
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch09-user --description "Deploy/operate ch-09" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch09-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch09-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch09-user ready"