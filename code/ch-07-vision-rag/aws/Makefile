.PHONY: iam build-setup serving-image deploy endpoint seed invoke status delete

STACK ?= ch07-vision-rag

# The GPU serving image is built by a second copy of build.yaml; the template is
# parameterised by RepoName, so the same file makes both repositories and projects.
SERVING_REPO ?= ch07-kyc-serving
SERVING_BUILD_STACK := $(SERVING_REPO)-build
ENDPOINT_NAME ?= ch07-kyc-face
INSTANCE_TYPE ?= ml.g4dn.xlarge

# One-time: ECR repository + CodeBuild project + source bucket.
build-setup:
	aws cloudformation deploy --template-file build.yaml \
		--stack-name $(SERVING_BUILD_STACK) --capabilities CAPABILITY_IAM \
		--parameter-overrides RepoName=$(SERVING_REPO)

# Zip the build context, upload it, and build the image in CodeBuild (native
# amd64, in the cloud, no laptop cross-build, no multi-GB push from home).
# The GPU serving image: the same staging trick, with the endpoint's Flask app in
# place of the Lambda handlers. CUDA torch makes this image several GB, so the build
# takes noticeably longer than the Lambda one.
serving-image:
	@rm -rf /tmp/ch07-serving-ctx && mkdir -p /tmp/ch07-serving-ctx
	cp serving/Dockerfile serving/buildspec.yml serving/serve.py serving/entrypoint.sh /tmp/ch07-serving-ctx/
	cp ../src/face_embedder.py ../src/face_explainer.py ../src/face_spec.py ../src/kycstore.py /tmp/ch07-serving-ctx/
	cd /tmp/ch07-serving-ctx && zip -qr /tmp/ch07-serving.zip .
	@BUCKET=$$(aws cloudformation describe-stacks --stack-name $(SERVING_BUILD_STACK) \
		--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" --output text); \
		aws s3 cp /tmp/ch07-serving.zip s3://$$BUCKET/src.zip; \
		echo "started build: $$(aws codebuild start-build --project-name $(SERVING_REPO)-build --query build.id --output text)"

# The asynchronous GPU endpoint. Reads everything it needs from the app stack outputs.
# It bills per instance-hour while an instance is up, so `make delete` matters here.
# Read one stack output: $(call out,STACK,KEY)
out = $$(aws cloudformation describe-stacks --stack-name $(1) \
        --query "Stacks[0].Outputs[?OutputKey=='$(2)'].OutputValue" --output text)

endpoint:
	@IMAGE_URI=$(call out,$(SERVING_BUILD_STACK),ImageUri) \
	 SAGEMAKER_ROLE_ARN=$(call out,$(STACK),SageMakerRoleArn) \
	 IMAGES_BUCKET=$(call out,$(STACK),ImagesBucket) \
	 PGHOST=$(call out,$(STACK),DatabaseEndpoint) \
	 PGPORT=$(call out,$(STACK),DatabasePort) \
	 PGUSER=$(call out,$(STACK),DBUser) \
	 PGDATABASE=$(call out,$(STACK),DBName) \
	 SNS_TOPIC_ARN=$(call out,$(STACK),SnsTopicArn) \
	 ENDPOINT_NAME=$(ENDPOINT_NAME) INSTANCE_TYPE=$(INSTANCE_TYPE) \
	 uv run deploy.py

# Upload identity photos and probe selfies. The probes/ and impostors/ objects are
# inert; the enrolled/ ones fire the enrolment trigger, which is the whole enrolment
# path: S3 event -> Lambda -> asynchronous endpoint -> Aurora -> SNS -> status table.
FACES ?= ../data/generated/faces
SUBJECTS ?= 8
seed:
	@B=$(call out,$(STACK),ImagesBucket); \
	for s in $$(ls $(FACES)/enrolled | head -$(SUBJECTS)); do \
		aws s3 cp $(FACES)/probe/$$s/selfie.jpg s3://$$B/probes/$$s/selfie.jpg --only-show-errors; \
		aws s3 cp $(FACES)/impostors/$$s/selfie.jpg s3://$$B/impostors/$$s/selfie.jpg --only-show-errors; \
	done; \
	for s in $$(ls $(FACES)/enrolled | head -$(SUBJECTS)); do \
		aws s3 cp $(FACES)/enrolled/$$s/id.jpg s3://$$B/enrolled/$$s/id.jpg --only-show-errors; \
	done; \
	 echo "seeded $(SUBJECTS) subjects"

# One verification through the endpoint: submit to S3, then wait on the status row the
# notification subscriber writes. VIA=s3 waits on the output object instead.
KEY ?= probes/subject_000/selfie.jpg
CLAIM ?= subject_000
VIA ?= status
invoke:
	@IMAGES_BUCKET=$(call out,$(STACK),ImagesBucket) ENDPOINT_NAME=$(ENDPOINT_NAME) \
	 STATUS_TABLE=$(call out,$(STACK),StatusTable) \
	 uv run invoke.py --key $(KEY) --claim $(CLAIM) --via $(VIA)

# What the subscriber has written so far: one row per submitted verification.
status:
	@aws dynamodb scan --table-name $(call out,$(STACK),StatusTable) \
	  --query "Items[].{id:verification_id.S,status:status.S,at:updated_at.S}" --output table

# Deploy the app stack: the express Aurora cluster, the images bucket, the enrolment
# trigger, the results topic, and the status table its subscriber writes.
deploy:
	sam build
	sam deploy --stack-name $(STACK) --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
		--parameter-overrides EndpointName=$(ENDPOINT_NAME) --no-confirm-changeset \
		--resolve-s3

# One command: CloudFormation will not delete a non-empty S3 bucket, so empty the
# images + source buckets first, then remove both stacks.
delete:
	-@IMAGES_BUCKET=$(call out,$(STACK),ImagesBucket) ENDPOINT_NAME=$(ENDPOINT_NAME) \
	  uv run deploy.py --delete || true
	-@B=$(call out,$(STACK),ImagesBucket); \
	  [ -n "$$B" ] && [ "$$B" != "None" ] && aws s3 rm "s3://$$B" --recursive || true
	sam delete --stack-name $(STACK) --no-prompts
	-@S=$(call out,$(SERVING_BUILD_STACK),SourceBucket); \
	  [ -n "$$S" ] && [ "$$S" != "None" ] && aws s3 rm "s3://$$S" --recursive || true
	aws cloudformation delete-stack --stack-name $(SERVING_BUILD_STACK)

# -------------------------------------------------------------------------------
# iam: provision THIS chapter's deploy identity, when you start the chapter
# -------------------------------------------------------------------------------
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	uv run iam/check_policy.py iam/deploy.json
	@aws iam create-role --role-name ch07-user --description "Deploy/operate ch-07" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch07-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch07-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch07-user ready"