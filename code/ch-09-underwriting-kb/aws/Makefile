.PHONY: iam secret build-setup source image deploy gen seed url ask cases teardown

REGION := us-east-1
STACK := ch09-underwriting-kb
BUILD_STACK := ch09-app-build
REPO := ch09-app
SECRET := ch09-os-master
MASTER := kbadmin

output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(STACK) --query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)
build_output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(BUILD_STACK) --query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)

ENDPOINT = $(call output,DomainEndpoint)
SOURCE_BUCKET = $(call build_output,SourceBucket)
IMAGE = $(call build_output,ImageUri)

Q ?= How is DTI assessed for a grocery business?
DEAL ?= A logistics firm seeks a 500,000 US dollar working-capital loan for six months.

# The store env that points the shared code at the domain instead of localhost.
# The master password is read from the secret inside each recipe, never expanded
# by make, so its punctuation cannot break word-splitting.
OSCONN = STORE=opensearch OPENSEARCH_HOST=$(ENDPOINT) OPENSEARCH_PORT=443 OPENSEARCH_SSL=1 \
	OPENSEARCH_USER=$(MASTER) BEDROCK_REGION=$(REGION) PYTHONPATH=src
GETPW = aws secretsmanager get-secret-value --secret-id $(SECRET) --region $(REGION) --query SecretString --output text

# -------------------------------------------------------------------------------
# deploy order: build-setup -> source -> image -> secret+deploy -> seed
# -------------------------------------------------------------------------------
# secret: the master password the template resolves into the domain and the app.
secret:
	@aws secretsmanager describe-secret --secret-id $(SECRET) --region $(REGION) >/dev/null 2>&1 || \
		aws secretsmanager create-secret --name $(SECRET) --region $(REGION) --secret-string \
		"$$(aws secretsmanager get-random-password --region $(REGION) --exclude-punctuation \
		--password-length 20 --query RandomPassword --output text)aA1!" >/dev/null
	@echo "secret $(SECRET) ready"

# build-setup: ECR + CodeBuild + a source bucket for the app image
build-setup:
	aws cloudformation deploy --region $(REGION) --template-file app-build.yaml \
		--stack-name $(BUILD_STACK) --capabilities CAPABILITY_IAM

# source: zip the app build context (Dockerfile + src + app) and upload it
source:
	cd .. && rm -f /tmp/ch09-app.zip && \
		zip -qr /tmp/ch09-app.zip Dockerfile src app \
		-x '*/__pycache__/*' '*/.ruff_cache/*' '*/.venv/*' '*/data/*'
	aws s3 cp /tmp/ch09-app.zip s3://$(SOURCE_BUCKET)/src.zip --region $(REGION)

# image: build the app image in the cloud (native amd64) and push it to ECR
image:
	@ID=$$(aws codebuild start-build --region $(REGION) --project-name $(REPO)-build \
		--query 'build.id' --output text); \
		echo "build $$ID"; \
		while STATUS=$$(aws codebuild batch-get-builds --region $(REGION) --ids $$ID \
			--query 'builds[0].buildStatus' --output text); \
			[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
			echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]

# deploy: the domain + the app (container Lambda via the Web Adapter + API Gateway)
deploy: secret
	aws cloudformation deploy --region $(REGION) --stack-name $(STACK) \
		--template-file template.yaml --no-fail-on-empty-changeset \
		--capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
		--parameter-overrides ImageUri=$(IMAGE) MasterUserName=$(MASTER)

# gen + seed: build the synthetic corpus, then embed it into the domain with the
# same code that runs locally (embeddings come from Bedrock, indexing from home).
gen:
	cd .. && uv run etl/gen_memos.py --out data/generated/memos --count 200 --seed 7 --messy

seed:
	cd .. && PW=$$($(GETPW)); $(OSCONN) OPENSEARCH_PASSWORD="$$PW" \
		uv run etl/embed_memos.py --memos data/generated/memos

url:
	@echo "Dashboards: $(call output,DashboardsURL)"
	@echo "App:        $(call output,AppUrl)"
	@echo "sign in to Dashboards as $(MASTER); password is in secret $(SECRET)"

ask:
	cd .. && PW=$$($(GETPW)); $(OSCONN) OPENSEARCH_PASSWORD="$$PW" \
		uv run src/retrieve.py ask --query "$(Q)"

cases:
	cd .. && PW=$$($(GETPW)); $(OSCONN) OPENSEARCH_PASSWORD="$$PW" \
		uv run src/retrieve.py cases --deal "$(DEAL)"

teardown:
	aws cloudformation delete-stack --region $(REGION) --stack-name $(STACK)
	aws cloudformation wait stack-delete-complete --region $(REGION) --stack-name $(STACK)
	-aws s3 rm s3://$(SOURCE_BUCKET) --recursive --region $(REGION)
	aws cloudformation delete-stack --region $(REGION) --stack-name $(BUILD_STACK)
	aws cloudformation wait stack-delete-complete --region $(REGION) --stack-name $(BUILD_STACK)
	-aws secretsmanager delete-secret --secret-id $(SECRET) --region $(REGION) --force-delete-without-recovery
	@echo "torn down"

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