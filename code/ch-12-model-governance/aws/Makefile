.PHONY: iam deploy image serve flip score govern verify-artifacts artifacts round down teardown

# the whole AWS round in one command (run make iam once first, as the account admin): the
# stack (AppConfig + the CodeBuild images + two SageMaker endpoints + the Fargate gateway),
# a batch of applications, the champion/challenger split through the running gateway, and
# the governance records (model cards, registry, lineage) uploaded to SageMaker. make
# teardown when done.
round:
	$(MAKE) deploy
	$(MAKE) -C .. apps
	$(MAKE) score
	@# The rollout is what the chapter is about, and it sat outside the round:
	@# nothing moved the percentage and scored again in the same run.
	$(MAKE) flip PCT=100
	$(MAKE) govern
	$(MAKE) verify-artifacts

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
	@# The endpoints load their model artifacts from the chapter's bucket, so
	@# the artifacts go up before the deploy that creates them. The registry
	@# names the same objects: a registered version that pointed somewhere the
	@# endpoint does not read would say nothing about what is serving.
	@$(MAKE) --no-print-directory artifacts
	@# The deploy pins the tag the build produced. Without it the models keep
	@# the digest :latest pointed at when they were first created.
	$(DEPLOY) RolloutPct=$(PCT) AllowedCidr=$(MYIP) Serving=true DesiredCount=1 \
		ImageTag=$$(cat /tmp/ch12-image.tag)
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
		TAG=img-$$(shasum -a 256 /tmp/ch12-src.zip | cut -c1-12); \
		echo $$TAG > /tmp/ch12-image.tag; \
		ID=$$(aws codebuild start-build-batch --project-name ch12-models-build \
			--environment-variables-override name=TAG,value=$$TAG,type=PLAINTEXT \
			--query 'buildBatch.id' --output text); \
		echo "batch $$ID"; \
		while STATUS=$$(aws codebuild batch-get-build-batches --ids $$ID --query 'buildBatches[0].buildBatchStatus' --output text); \
			[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
			echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]; \
			echo "pushed $$(cat /tmp/ch12-image.tag)"

# serve: (re)deploy just the serving layer, gateway running. Use after `make image` when
# the build infrastructure and AppConfig are already up.
serve:
	$(DEPLOY) RolloutPct=$(PCT) AllowedCidr=$(MYIP) Serving=true DesiredCount=1
	@echo "serving layer up (rollout $(PCT)%)"

# flip: widen the rollout. A new configuration version and deployment; the ECS service
# and its running task are untouched. Then re-score to show the shift.
flip:
	@# ImageTag too: this is a deploy, and leaving it out takes the parameter
	@# back to its default and the models back to whatever :latest meant when
	@# they were created.
	$(DEPLOY) RolloutPct=$(PCT) AllowedCidr=$(MYIP) Serving=true DesiredCount=1 \
		ImageTag=$$(cat /tmp/ch12-image.tag)
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

# govern: upload each model's card, registry entry, and lineage to SageMaker governance
# (idempotent; reruns update in place).
# The registry points at model artifacts in S3, and nothing used to put them
# there: the keys in governance/models.json were filled by hand once, in the
# account's default SageMaker bucket, so a reader running this cold registered
# packages against objects that did not exist. These are ch-02's trained models,
# packaged and uploaded to this chapter's own bucket.
CH02 := ../../ch-02-mlops/runs-local
artifacts:
	@test -f $(CH02)/model/scorecard.joblib || { \
		echo "no champion model at $(CH02)/model/scorecard.joblib"; \
		echo "train it first: make -C ../../ch-02-mlops train"; exit 1; }
	@test -f $(CH02)/model-challenger/challenger.cbm || { \
		echo "no challenger model at $(CH02)/model-challenger/challenger.cbm"; \
		echo "train it first: make -C ../../ch-02-mlops train-challenger"; exit 1; }
	@BUCKET=$$(aws cloudformation describe-stacks --stack-name $(STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" \
			--output text); \
			WORK=$$(mktemp -d); \
			tar -czf $$WORK/champion.tar.gz -C $(CH02)/model scorecard.joblib; \
			tar -czf $$WORK/challenger.tar.gz -C $(CH02)/model-challenger challenger.cbm; \
			aws s3 cp $$WORK/champion.tar.gz s3://$$BUCKET/ch12/champion-pl/model.tar.gz; \
			aws s3 cp $$WORK/challenger.tar.gz s3://$$BUCKET/ch12/challenger-cb/model.tar.gz; \
			echo "artifacts uploaded to s3://$$BUCKET/ch12/"

# The endpoint and the registered package are configured from the same stack
# output and the same key, which makes them the same URI but not, on its own,
# the same bytes: artifacts is run on every deploy, so a key can be overwritten
# between the endpoint reading it and the package recording it. This compares
# both URIs and the ETag of the object each one names.
verify-artifacts:
	@# The model names are CloudFormation's, not ours: the template does not set
	@# ModelName, so they come out as ChampionModel-<suffix>. The logical id is
	@# the stable handle, so ask the stack what the physical name is.
	@for pair in "ChampionModel credit-scorecard" "ChallengerModel credit-challenger"; do \
		set -- $$pair; \
		NAME=$$(aws cloudformation describe-stack-resource --stack-name $(STACK) \
		--logical-resource-id "$$1" \
		--query 'StackResourceDetail.PhysicalResourceId' --output text); \
		NAME=$${NAME##*/}; \
		MODEL=$$(aws sagemaker describe-model --model-name "$$NAME" \
		--query 'PrimaryContainer.ModelDataUrl' --output text); \
		ARN=$$(aws sagemaker list-model-packages --model-package-group-name "$$2" \
		--query 'ModelPackageSummaryList[0].ModelPackageArn' --output text); \
		PKG=$$(aws sagemaker describe-model-package --model-package-name "$$ARN" \
		--query 'InferenceSpecification.Containers[0].ModelDataUrl' --output text); \
		if [ "$$MODEL" != "$$PKG" ]; then \
			echo "$$1: the endpoint and the registry name different artifacts"; \
			echo "  endpoint: $$MODEL"; \
			echo "  registry: $$PKG"; \
			exit 1; \
		fi; \
		BUCKET=$$(echo "$$MODEL" | sed -E 's|s3://([^/]+)/.*|\1|'); \
		KEY=$$(echo "$$MODEL" | sed -E 's|s3://[^/]+/||'); \
		ETAG=$$(aws s3api head-object --bucket "$$BUCKET" --key "$$KEY" \
			--query ETag --output text | tr -d '"'); \
			RECORDED=$$(aws sagemaker describe-model-package --model-package-name "$$ARN" \
			--query 'CustomerMetadataProperties.model_artifact_etag' --output text); \
			if [ "$$ETAG" != "$$RECORDED" ]; then \
				echo "$$2: the artifact changed since it was registered"; \
				echo "  registered: $$RECORDED"; \
				echo "  in the bucket now: $$ETAG"; \
				exit 1; \
			fi; \
		echo "$$NAME and $$2 both name s3://$$BUCKET/$$KEY, ETag $$ETAG as registered"; \
	done

govern:
	@BUCKET=$$(aws cloudformation describe-stacks --stack-name $(STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" \
			--output text); \
			cd .. && ARTIFACT_BUCKET=$$BUCKET IMAGE_TAG=$$(cat /tmp/ch12-image.tag) \
			uv run src/governance.py --upload --region $(REGION)

# down: park the gateway at 0. The stack stays up but costs nothing at rest.
down:
	aws ecs update-service --cluster ch12-gateway --service ch12-gateway --desired-count 0 >/dev/null
	@echo "gateway parked at 0"

# teardown: empty the source bucket first (S3 will not delete a non-empty bucket, so the
# stack delete would otherwise fail), turn off AppConfig deletion protection (it blocks
# deleting a profile or environment used within the last hour, which the score step just
# did), then delete the stack and wait for it to finish.
teardown:
	-@BUCKET=$$(aws cloudformation describe-stacks --stack-name $(STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" --output text 2>/dev/null); \
		if [ -n "$$BUCKET" ] && [ "$$BUCKET" != "None" ]; then \
			echo "emptying $$BUCKET"; aws s3 rm s3://$$BUCKET --recursive; fi
	-aws appconfig update-account-settings --deletion-protection Enabled=false >/dev/null
	aws cloudformation delete-stack --stack-name $(STACK)
	aws cloudformation wait stack-delete-complete --stack-name $(STACK)
	@echo "$(STACK) torn down"
