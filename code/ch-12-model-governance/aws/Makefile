.PHONY: iam deploy image serve flip score down teardown

# The governance round on AWS, one stack end to end: a CodeBuild batch build produces the
# three images, AppConfig drives the rollout, the AppConfig agent sidecar on a Fargate
# gateway reads it, and two SageMaker endpoints serve the models. Account values are
# derived; override on the command line.
REGION ?= us-east-1
ACCOUNT := $(shell aws sts get-caller-identity --query Account --output text)
STACK := ch12-governance
PCT ?= 20
SAMPLE ?= 60
# The gateway port opens to this address only; default to the caller's public IP.
MYIP := $(shell curl -s https://checkip.amazonaws.com)/32
DEPLOY = aws cloudformation deploy --template-file template.yaml --stack-name $(STACK) --capabilities CAPABILITY_NAMED_IAM --parameter-overrides

# -------------------------------------------------------------------------------
# iam: provision this chapter's deploy identity (run once, as the account admin)
# -------------------------------------------------------------------------------
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch12-user --description "Deploy/operate ch-12" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch12-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch12-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch12-user ready; waiting for the policy to propagate"; sleep 10

# -------------------------------------------------------------------------------
# deploy: the whole stack in two passes, images built (in parallel) in between
# -------------------------------------------------------------------------------
# Pass 1 brings up the build infrastructure and AppConfig (Serving=false). The batch
# build pushes the three images. Pass 2 adds the endpoints and the running gateway
# (Serving=true), which need the images to exist. Waits between passes let each settle.
deploy:
	$(DEPLOY) RolloutPct=$(PCT) AllowedCidr=$(MYIP) Serving=false DesiredCount=0
	@echo "build infrastructure up; waiting for propagation"; sleep 10
	@$(MAKE) --no-print-directory image
	@echo "images pushed; waiting for propagation"; sleep 10
	$(DEPLOY) RolloutPct=$(PCT) AllowedCidr=$(MYIP) Serving=true DesiredCount=1
	@echo "stack $(STACK) ready (rollout $(PCT)%, gateway open to $(MYIP))"

# image: build the three images in parallel (CodeBuild batch -> ECR), natively on amd64.
# Stage the build contexts side by side (chapter 2's scorecard + challenger, and this
# chapter's gateway), upload, run the batch build, and poll to SUCCEEDED.
image:
	rm -rf /tmp/ch12-build && mkdir -p /tmp/ch12-build/gateway
	cp -R ../../ch-02-mlops/src/scorecard ../../ch-02-mlops/src/challenger /tmp/ch12-build/
	cp -R ../src ../app ../Dockerfile /tmp/ch12-build/gateway/
	cp buildspec.yml /tmp/ch12-build/
	rm -f /tmp/ch12-src.zip && cd /tmp/ch12-build && zip -qr /tmp/ch12-src.zip . -x '*__pycache__*'
	@BUCKET=$$(aws cloudformation describe-stacks --stack-name $(STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" --output text); \
		aws s3 cp /tmp/ch12-src.zip s3://$$BUCKET/src.zip; \
		ID=$$(aws codebuild start-build-batch --project-name ch12-models-build --query 'buildBatch.id' --output text); \
		echo "batch $$ID"; \
		while STATUS=$$(aws codebuild batch-get-build-batches --ids $$ID --query 'buildBatches[0].buildBatchStatus' --output text); \
			[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
			echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]

# serve: (re)deploy just the serving layer, gateway running. Use after `make image` when
# the build infrastructure and AppConfig are already up.
serve:
	$(DEPLOY) RolloutPct=$(PCT) AllowedCidr=$(MYIP) Serving=true DesiredCount=1
	@echo "serving layer up (rollout $(PCT)%)"

# flip: widen the rollout. A new configuration version and deployment; the ECS service
# and its running task are untouched. Then re-score to show the shift.
flip:
	$(DEPLOY) RolloutPct=$(PCT) AllowedCidr=$(MYIP) Serving=true DesiredCount=1
	@echo "rollout -> $(PCT)%; waiting for the configuration to propagate"; sleep 15
	@$(MAKE) --no-print-directory score

# score: POST a sample of applications through the running gateway and tally the split.
score:
	@aws ecs wait services-stable --cluster ch12-gateway --services ch12-gateway
	@TASK=$$(aws ecs list-tasks --cluster ch12-gateway --service-name ch12-gateway --query 'taskArns[0]' --output text); \
		ENI=$$(aws ecs describe-tasks --cluster ch12-gateway --tasks $$TASK \
		--query "tasks[0].attachments[0].details[?name=='networkInterfaceId']|[0].value" --output text); \
		IP=$$(aws ec2 describe-network-interfaces --network-interface-ids $$ENI \
		--query 'NetworkInterfaces[0].Association.PublicIp' --output text); \
		echo "gateway http://$$IP:8080"; \
		head -$(SAMPLE) ../data/applications.jsonl > /tmp/ch12-sample.jsonl; \
		GATEWAY_URL=http://$$IP:8080/score uv run ../etl/score.py /tmp/ch12-sample.jsonl

# down: park the gateway at 0. The stack stays up but costs nothing at rest.
down:
	aws ecs update-service --cluster ch12-gateway --service ch12-gateway --desired-count 0 >/dev/null
	@echo "gateway parked at 0"

# teardown: empty the source bucket first (S3 will not delete a non-empty bucket, so the
# stack delete would otherwise fail), then delete the stack and wait for it to finish.
teardown:
	-@BUCKET=$$(aws cloudformation describe-stacks --stack-name $(STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" --output text 2>/dev/null); \
		if [ -n "$$BUCKET" ] && [ "$$BUCKET" != "None" ]; then \
			echo "emptying $$BUCKET"; aws s3 rm s3://$$BUCKET --recursive; fi
	aws cloudformation delete-stack --stack-name $(STACK)
	aws cloudformation wait stack-delete-complete --stack-name $(STACK)
	@echo "$(STACK) torn down"