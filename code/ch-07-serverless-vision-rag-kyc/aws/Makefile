.PHONY: build-setup image deploy delete

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

delete:
	cd vision-rag && sam delete --stack-name $(STACK) --no-prompts
	aws cloudformation delete-stack --stack-name $(BUILD_STACK)