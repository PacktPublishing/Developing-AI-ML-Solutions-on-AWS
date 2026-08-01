.PHONY: lint deploy image scale-up scale-down decide teardown

REGION ?= us-east-1
STACK := realtime-scoring-ch04

output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(STACK) \
	--query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)

lint:
	uvx cfn-lint template.yaml
	rain fmt --verify template.yaml

deploy:
	sam deploy --no-confirm-changeset --no-fail-on-empty-changeset

# build the service image in the cloud and push it to the stack's ECR repo
image:
	$(eval BUCKET := $(call output,StagingBucketName))
	cd .. && rm -f /tmp/ch04-app-src.zip && \
		zip -q -j /tmp/ch04-app-src.zip src/Dockerfile src/app.py src/scorecard.py \
		aws/image/buildspec.yml
	aws s3 cp /tmp/ch04-app-src.zip s3://$(BUCKET)/ch04/source/app.zip
	@ID=$$(aws codebuild start-build --region $(REGION) --project-name ch04-decision-build \
		--query 'build.id' --output text); \
	echo "build $$ID"; \
	while STATUS=$$(aws codebuild batch-get-builds --region $(REGION) --ids $$ID \
		--query 'builds[0].buildStatus' --output text); \
		[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
		echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]

# start one task behind the ALB (the stack deploys at desired count 0)
scale-up:
	aws ecs update-service --region $(REGION) --cluster ch04-decision \
		--service decision-service --desired-count 1 >/dev/null
	aws ecs wait services-stable --region $(REGION) --cluster ch04-decision \
		--services decision-service
	@echo "service up: $(call output,ServiceUrl)"

scale-down:
	aws ecs update-service --region $(REGION) --cluster ch04-decision \
		--service decision-service --desired-count 0 >/dev/null

# one application through the real endpoint
decide:
	curl -s $(call output,ServiceUrl)/decision -H 'content-type: application/json' -d '{ \
	  "application_id": "app-1042", "age": 34, "monthly_income": 4200, \
	  "requested_amount": 8000, "dti": 0.20, "utilization": 0.15 }'
	@echo

teardown:
	-aws ecs update-service --region $(REGION) --cluster ch04-decision \
		--service decision-service --desired-count 0
	-aws s3 rm --recursive s3://$(call output,StagingBucketName)
	sam delete --no-prompts --region $(REGION)