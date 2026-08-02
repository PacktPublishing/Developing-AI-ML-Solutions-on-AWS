#!/bin/bash
set -e

# The Data API region; the helper reads it at call time.
export REDSHIFT_REGION="${REDSHIFT_REGION:-us-east-1}"

# Light terminal theme: white background, dark text, book-friendly.
THEME='{"background":"#ffffff","foreground":"#1f2937","cursor":"#374151","cursorAccent":"#ffffff","selectionBackground":"#bfdbfe","black":"#1f2937","brightBlack":"#6b7280","red":"#dc2626","brightRed":"#ef4444","green":"#16a34a","brightGreen":"#22c55e","yellow":"#d97706","brightYellow":"#f59e0b","blue":"#2563eb","brightBlue":"#3b82f6","magenta":"#7c3aed","brightMagenta":"#8b5cf6","cyan":"#0891b2","brightCyan":"#06b6d4","white":"#f9fafb","brightWhite":"#ffffff"}'

# ttyd runs the restricted launcher: SHELL points at noshell so a user cannot
# drop out of Claude Code into a raw shell; -W makes the terminal writable;
# -a forwards ?arg=<user> from the URL to launch-claude as $1.
exec env SHELL=/usr/local/bin/noshell \
  ttyd -p 7681 -W -a -t "theme=$THEME" /usr/local/bin/launch-claude
