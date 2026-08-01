.PHONY: up seed run query down clean lint

export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1

up:
	docker compose up -d --wait

seed:
	uv run data/generate_portfolio.py --rows 20000 --seed 5
	uv run steps/seed_warehouse.py

# the batch run, step by step: shortlist -> score -> decide -> apply
run:
	uv run steps/shortlist.py
	uv run steps/score.py
	uv run steps/decide.py
	uv run steps/apply_limits.py

query:
	docker compose exec warehouse psql -h 127.0.0.1 -p 5433 -U analyst -d portfolio \
		-c "SELECT ROUND(AVG(current_limit)) AS avg_limit, COUNT(*) FROM customers"

down:
	docker compose down -v

clean: down
	rm -rf data/portfolio.csv data/feature_spec.json catboost_info

lint:
	uvx ruff format data/*.py steps/*.py
	uvx ruff check data/*.py steps/*.py