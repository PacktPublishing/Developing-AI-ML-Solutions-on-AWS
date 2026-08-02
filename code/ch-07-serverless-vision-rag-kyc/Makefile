.PHONY: diagrams lint

lint:
	uvx ruff check .

# Regenerate the architecture figures from diagrams/yaml/ with a pinned awsdac.
# AWS icons are fetched at render time; custom icons live in the shared
# assets/diagrams/ (Strands, Lambda coin, local-stack tools). Icon PNGs are
# re-rendered from their SVG sources at ICON_HEIGHT (needs librsvg's rsvg-convert).
# WIDTH sets output pixel width (0 = awsdac native; raise it for print).
AWSDAC_VERSION := v0.22.4
AWSDAC := $(shell go env GOPATH)/bin/awsdac
WIDTH ?= 0
ICON_HEIGHT ?= 500
diagrams:
	@test -x "$(AWSDAC)" || go install github.com/awslabs/diagram-as-code/cmd/awsdac@$(AWSDAC_VERSION)
	@for s in ../../assets/diagrams/icons/*.svg; do \
		case "$$(basename $$s)" in fa-*) continue;; esac; \
			rsvg-convert -h $(ICON_HEIGHT) "$$s" -o "../../assets/diagrams/icons/$$(basename $$s .svg).png"; \
		done
	@../../assets/diagrams/icons/render-fa.sh >/dev/null
	@mkdir -p diagrams/png
	@for f in diagrams/yaml/*.yaml; do \
		echo "  $$f"; \
		"$(AWSDAC)" -f "$$f" -o "diagrams/png/$$(basename $$f .yaml).png" --width $(WIDTH); \
	done