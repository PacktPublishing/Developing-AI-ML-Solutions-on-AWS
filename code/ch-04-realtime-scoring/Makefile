.PHONY: up decide refer decline log sm-local kube-up kube-decide kube-down down clean lint test

export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1

COMPOSE := docker compose -f local/decision-service.yml

# the decision service, local: DynamoDB plus the two models as concurrent SageMaker local
# endpoints (each on its own serving_port, 8081/8082) with the gateway ensembling them. Runs
# in the foreground; score it from another shell (make decide/refer/decline), stop with Ctrl-C.
up:
	docker build -q -t ch04-scorecard-model src/models/scorecard
	docker build -q -t ch04-challenger-model src/models/challenger
	$(COMPOSE) up -d decisions createtable
	uv run src/serve_local.py

# one clean application: expect APPROVE (both models, and the average, below cutoff)
decide:
	curl -s localhost:8080/decision -H 'content-type: application/json' \
	  -d '{"application_id":"app-1042","age":40,"monthly_income":6000,"requested_amount":5000,"dti":0.10,"utilization":0.10}'
	@echo

# nudge into the gray band: expect REFER (ensemble PD between the cutoffs)
refer:
	curl -s localhost:8080/decision -H 'content-type: application/json' \
	  -d '{"application_id":"app-1043","age":34,"monthly_income":4200,"requested_amount":8000,"dti":0.20,"utilization":0.20}'
	@echo

# fail KYC: expect DECLINE with the reason code, whatever the score
decline:
	curl -s localhost:8080/decision -H 'content-type: application/json' \
	  -d '{"application_id":"app-1044","age":34,"monthly_income":4200,"requested_amount":8000,"dti":0.20,"utilization":0.15,"kyc_passed":false}'
	@echo

# the audit trail: one row per application, the ensemble and each model's PD
log:
	aws dynamodb scan --endpoint-url http://localhost:8000 \
		--table-name decision_log --select COUNT --query Count

# Option 2, SageMaker local mode: the same ModelBuilder code as the cloud, Mode.LOCAL_CONTAINER,
# the container on this machine. The image is built locally (not in ECR), and ModelBuilder
# always pulls the serving image, so this runs in the no-account mode (SM_OFFLINE=1): the SDK's
# account, role, and image-pull calls are stubbed (src/sagemaker_offline.py). The credentialed
# counterpart is the cloud endpoint (make -C aws deploy). MODEL=scorecard|challenger.
MODEL ?= scorecard
sm-local:
	docker build -q -t ch04-$(MODEL)-model src/models/$(MODEL)
	SM_OFFLINE=1 MODE=local MODEL_IMAGE=ch04-$(MODEL)-model:latest \
		SAGEMAKER_ROLE_ARN=arn:aws:iam::000000000000:role/local \
		uv run --no-project src/deploy.py

# the same gateway and models on Kubernetes: a k3d cluster (k3s in docker), the
# three images imported, the ensemble manifest applied. Needs k3d and kubectl.
kube-up:
	@command -v k3d >/dev/null 2>&1 || { echo "k3d is not installed: https://k3d.io/#installation"; exit 1; }
	@command -v kubectl >/dev/null 2>&1 || { echo "kubectl is not installed"; exit 1; }
	k3d cluster create decision --port "8085:80@loadbalancer" --wait
	docker build -q -t ch04-scorecard-model src/models/scorecard
	docker build -q -t ch04-challenger-model src/models/challenger
	docker build -q -t ch04-gateway src/gateway
	k3d image import -c decision ch04-scorecard-model ch04-challenger-model ch04-gateway
	kubectl apply -f k8s/ensemble.yaml
	kubectl rollout status deployment/gateway --timeout=120s

kube-decide:
	curl -s --retry 10 --retry-delay 2 --retry-all-errors \
	  localhost:8085/decision -H 'content-type: application/json' \
	  -d '{"application_id":"app-1042","age":40,"monthly_income":6000,"requested_amount":5000,"dti":0.10,"utilization":0.10}'
	@echo

kube-down:
	k3d cluster delete decision

down:
	-pkill -f serve_local.py 2>/dev/null || true
	-docker ps -aq --filter ancestor=ch04-scorecard-model:latest | xargs -r docker rm -f
	-docker ps -aq --filter ancestor=ch04-challenger-model:latest | xargs -r docker rm -f
	$(COMPOSE) down -v

clean: down
	docker rmi ch04-scorecard-model ch04-challenger-model ch04-gateway 2>/dev/null || true

lint:
	uvx ruff format src/
	uvx ruff check src/

# the gateway's decision rules, with no network and no AWS account
test:
	uv run --with pytest --with fastapi --with httpx --with boto3 --with pydantic pytest tests/ -q