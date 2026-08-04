.PHONY: iam build-setup image deploy delete

STACK ?= ch07-vision-rag
BUILD_STACK := $(STACK)-build

# One-time: ECR repository + CodeBuild project + source bucket.
build-setup:
	aws cloudformation deploy --template-file vision-rag/build.yaml \
		--stack-name $(BUILD_STACK) --capabilities CAPABILITY_IAM

# Zip the build context, upload it, and build the image in CodeBuild (native
# amd64, in the cloud — no laptop cross-build, no multi-GB push from home).
image:
	cd vision-rag/src && zip -qr /tmp/ch07-src.zip Dockerfile buildspec.yml *.py
	@BUCKET=$$(aws cloudformation describe-stacks --stack-name $(BUILD_STACK) \
		--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" --output text); \
		aws s3 cp /tmp/ch07-src.zip s3://$$BUCKET/src.zip; \
		echo "started build: $$(aws codebuild start-build --project-name $(STACK)-build --query build.id --output text)"

# Deploy the app stack (VPC, RDS + pgvector, two Lambdas, API Gateway) using the
# image CodeBuild pushed to ECR. Run `make image` and wait for it to finish first.
deploy:
	@IMAGE=$$(aws cloudformation describe-stacks --stack-name $(BUILD_STACK) \
		--query "Stacks[0].Outputs[?OutputKey=='ImageUri'].OutputValue" --output text); \
		aws cloudformation deploy --template-file vision-rag/template.yaml \
		--stack-name $(STACK) --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
		--parameter-overrides ImageUri=$$IMAGE

# One command: CloudFormation will not delete a non-empty S3 bucket, so empty the
# images + source buckets first, then remove both stacks.
delete:
	-@B=$$(aws cloudformation describe-stacks --stack-name $(STACK) \
		--query "Stacks[0].Outputs[?OutputKey=='ImagesBucket'].OutputValue" --output text 2>/dev/null); \
		[ -n "$$B" ] && [ "$$B" != "None" ] && aws s3 rm "s3://$$B" --recursive || true
	cd vision-rag && sam delete --stack-name $(STACK) --no-prompts
	-@S=$$(aws cloudformation describe-stacks --stack-name $(BUILD_STACK) \
		--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" --output text 2>/dev/null); \
		[ -n "$$S" ] && [ "$$S" != "None" ] && aws s3 rm "s3://$$S" --recursive || true
	aws cloudformation delete-stack --stack-name $(BUILD_STACK)

# -------------------------------------------------------------------------------
# iam: provision THIS chapter's deploy identity, when you start the chapter
# Creates ch07-user from iam/deploy.json (least-privilege for ch-07 only) and trusts you to
# assume it — per chapter, not an account-wide up-front bootstrap. Re-run after
# editing the policy. Then add a [profile ch07] (role_arn + source_profile) and
# run cloud targets with AWS_PROFILE=ch07.
# -------------------------------------------------------------------------------
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch07-user --description "Deploy/operate ch-07" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch07-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch07-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch07-user ready"