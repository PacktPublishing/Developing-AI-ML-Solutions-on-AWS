.PHONY: up decide refer decline log kube-up kube-decide kube-down down clean lint diagrams

export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1

COMPOSE := docker compose -f local/decision-service.yml

up:
	$(COMPOSE) up -d --build --wait

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
	curl -s localhost:8085/decision -H 'content-type: application/json' -d '{ \
	  "application_id": "app-1042", "age": 34, "monthly_income": 4200, \
	  "requested_amount": 8000, "dti": 0.20, "utilization": 0.15 }'
	@echo

kube-down:
	k3d cluster delete decision

down:
	$(COMPOSE) down -v

clean: down
	docker rmi ch04-service 2>/dev/null || true

lint:
	uvx ruff format src/*.py
	uvx ruff check src/*.py
# Regenerate the architecture figures from diagrams/yaml/ with a pinned awsdac.
# Custom icons live in the shared assets/diagrams/; AWS icons fetched at render.
AWSDAC_VERSION := v0.22.4
AWSDAC := $(shell go env GOPATH)/bin/awsdac
WIDTH ?= 0
ICON_HEIGHT ?= 500
diagrams:
	@test -x "$(AWSDAC)" || go install github.com/awslabs/diagram-as-code/cmd/awsdac@$(AWSDAC_VERSION)
	@for s in ../../assets/diagrams/icons/*.svg; do \
		rsvg-convert -h $(ICON_HEIGHT) "$$s" -o "../../assets/diagrams/icons/$$(basename $$s .svg).png"; \
	done
	@mkdir -p diagrams/png
	@for f in diagrams/yaml/*.yaml; do \
		echo "  $$f"; \
		"$(AWSDAC)" -f "$$f" -o "diagrams/png/$$(basename $$f .yaml).png" --width $(WIDTH); \
	done