# Repository-level targets. The chapters are self-contained and keep their own
# Makefiles; what lives here is the work that is shared between them.
#
#   make icons   re-render assets/diagrams/icons/*.png from their SVG sources

ICON_DIR := assets/diagrams/icons
ICON_HEIGHT ?= 500
ICON_SVG := $(wildcard $(ICON_DIR)/*.svg)
ICON_PNG := $(ICON_SVG:.svg=.png)

.PHONY: icons
icons: $(ICON_PNG)

# The icons are shared, so they are rendered here rather than from a chapter.
# Each chapter's diagrams target used to run this loop itself, over every icon
# in the directory, which meant any chapter's run rewrote all sixteen and left
# whichever ones the pipeline renders differently showing as modified.
#
# Render to a temporary file and replace the PNG only when the bytes differ, so
# a run that changes nothing leaves the working tree clean. The mtime rule above
# is not enough on its own: git does not carry mtimes, so after a fresh clone
# make cannot tell which side is newer.
$(ICON_DIR)/%.png: $(ICON_DIR)/%.svg
	@command -v rsvg-convert >/dev/null || \
		{ echo "rsvg-convert not found; brew install librsvg"; exit 1; }
	@rsvg-convert -h $(ICON_HEIGHT) "$<" -o "$@.tmp"
	@if cmp -s "$@.tmp" "$@"; then \
		rm -f "$@.tmp"; \
	else \
		mv "$@.tmp" "$@"; echo "  rendered $@"; \
	fi