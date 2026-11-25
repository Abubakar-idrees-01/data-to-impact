#!/usr/bin/env bash
# Installs the daily digest timer for the current user (Linux, systemd). Run it yourself; undo with:
#   systemctl --user disable --now scout-digest.timer && rm ~/.config/systemd/user/scout-digest.*
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dest="$HOME/.config/systemd/user"
mkdir -p "$dest"
for f in scout-digest.service scout-digest.timer; do
  sed "s|%h/PATH/TO/data-to-impact/04-ai-agent|$here|g" "$here/scripts/systemd/$f" > "$dest/$f"
done
systemctl --user daemon-reload
systemctl --user enable --now scout-digest.timer
systemctl --user list-timers scout-digest.timer --no-pager
echo "Installed. Each morning a draft waits in .scout/pending/; approve it with: scout review"
