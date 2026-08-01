.PHONY: up decide refer decline log kube-up kube-decide kube-down eks-up eks-deploy eks-decide eks-down down clean lint

export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1

up:
	docker compose up -d --build --wait

# one clean application: expect APPROVE
decide:
	curl -s localhost:8080/decision -H 'content-type: application/json' -d '{ \
	  "application_id": "app-1042", "age": 34, "monthly_income": 4200, \
	  "requested_amount": 8000, "dti": 0.20, "utilization": 0.15 }'
	@echo

# nudge utilization into the gray band: expect REFER
refer:
	curl -s localhost:8080/decision -H 'content-type: application/json' -d '{ \
	  "application_id": "app-1043", "age": 34, "monthly_income": 4200, \
	  "requested_amount": 8000, "dti": 0.25, "utilization": 0.38 }'
	@echo

# fail KYC: expect DECLINE with the reason code, whatever the score
decline:
	curl -s localhost:8080/decision -H 'content-type: application/json' -d '{ \
	  "application_id": "app-1044", "age": 34, "monthly_income": 4200, \
	  "requested_amount": 8000, "dti": 0.20, "utilization": 0.15, \
	  "kyc_passed": false }'
	@echo

# the audit trail: decisions logged to DynamoDB, newest last
log:
	aws dynamodb scan --endpoint-url http://localhost:8000 \
		--table-name decision_log --select COUNT --query Count

# the same image on Kubernetes: a k3d cluster (k3s in docker), the image
# imported, the manifests applied. Needs k3d and kubectl installed.
kube-up:
	k3d cluster create decision --port "8085:80@loadbalancer" --wait
	docker build -q -t ch04-decision:local src/
	k3d image import -c decision ch04-decision:local
	kubectl apply -f k8s/decision-service.yaml
	kubectl rollout status deployment/decision-service --timeout=120s

kube-decide:
	curl -s --retry 10 --retry-delay 2 --retry-all-errors \
	  localhost:8085/decision -H 'content-type: application/json' -d '{ \
	  "application_id": "app-1042", "age": 34, "monthly_income": 4200, \
	  "requested_amount": 8000, "dti": 0.20, "utilization": 0.15 }'
	@echo

kube-down:
	k3d cluster delete decision

# the same manifests on EKS with a Fargate profile: create the cluster,
# swap the image for the ECR copy, apply. The control plane bills hourly —
# create, run, delete. Needs eksctl; the image comes from the aws/ stack.
eks-up:
	eksctl create cluster -f aws/eks/create-cluster.yaml

REPO = $(shell aws cloudformation describe-stacks --region us-east-1 \
	--stack-name realtime-scoring-ch04 \
	--query "Stacks[0].Outputs[?OutputKey=='ServiceRepoUri'].OutputValue" --output text)
eks-deploy:
	sed -e "s|ch04-decision:local|$(REPO):latest|" \
		-e "s|imagePullPolicy: Never|imagePullPolicy: Always|" \
		k8s/decision-service.yaml | kubectl apply -f -
	kubectl rollout status deployment/decision-service --timeout=300s

# no load balancer controller in scope: smoke-test through a port-forward
eks-decide:
	@kubectl port-forward svc/decision-service 8085:80 >/dev/null 2>&1 & \
	PF=$$!; sleep 3; \
	curl -s localhost:8085/decision -H 'content-type: application/json' -d '{ \
	"application_id": "app-1042", "age": 34, "monthly_income": 4200, \
	"requested_amount": 8000, "dti": 0.20, "utilization": 0.15 }'; \
	echo; kill $$PF

eks-down:
	eksctl delete cluster -f aws/eks/create-cluster.yaml

down:
	docker compose down -v

clean: down
	docker rmi ch-04-realtime-scoring-service ch04-decision:local 2>/dev/null || true

lint:
	uvx ruff format src/*.py
	uvx ruff check src/*.py