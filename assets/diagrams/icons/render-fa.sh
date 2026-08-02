#!/bin/sh
# Render the Font Awesome glyphs to PNG with colors from fa-colors.txt.
# The SVGs are upstream-pure; color is a render-time decision, so changing
# the palette means editing the map and re-running this.
set -e
cd "$(dirname "$0")"
HEIGHT="${ICON_HEIGHT:-500}"
grep -v '^#' fa-colors.txt | while read -r name color; do
  [ -n "$name" ] || continue
  printf 'path { fill: %s; }\n' "$color" >/tmp/fa-fill.css
  rsvg-convert -h "$HEIGHT" -s /tmp/fa-fill.css "$name.svg" -o "$name.png"
done
echo "rendered $(grep -cv '^#' fa-colors.txt) glyphs"
