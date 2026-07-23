.PHONY: sync build deploy teardown

STACK ?= underwriting-assistant-ch03
REGION ?= us-east-1

# The knowledge base has one source of truth, corpus_data.py at the chapter root.
# A Lambda package has to be self-contained, so the build copies that file into
# src/ instead of keeping a second copy under version control. src/corpus_data.py
# is gitignored for the same reason: one file to edit, no drift between the
# chapter's knowledge base and the deployed one.
sync:
	cp ../../corpus_data.py src/corpus_data.py

build: sync
	sam build --beta-features

deploy: build
	sam deploy --no-confirm-changeset --no-fail-on-empty-changeset \
		--capabilities CAPABILITY_IAM --region $(REGION)

teardown:
	sam delete --stack-name $(STACK) --region $(REGION) --no-prompts