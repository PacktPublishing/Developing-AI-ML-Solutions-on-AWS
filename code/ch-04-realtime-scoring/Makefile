.PHONY: up decide refer decline log down clean lint

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

down:
	docker compose down -v

clean: down
	docker rmi ch-04-realtime-scoring-service 2>/dev/null || true

lint:
	uvx ruff format app/*.py
	uvx ruff check app/*.py