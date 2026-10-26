#!/bin/sh
# Switch the running proxy between normal and emergency mode without restarting it (rate-limit counters are kept).
set -eu
MODE=${1:?usage: proxy_mode.sh normal|emergency}
DIR=${NII_PROXY_DIR:-/opt/nii/proxy}; PROXY=${NII_PROXY_CONTAINER:-release-proxy-1}
[ -f "$DIR/mode.$MODE.caddy" ] || { echo "unknown mode: $MODE"; exit 2; }
reload() { docker exec "$PROXY" caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile; }
cp -p "$DIR/mode.caddy" "$DIR/mode.caddy.previous"
cp "$DIR/mode.$MODE.caddy" "$DIR/mode.caddy.new" && mv "$DIR/mode.caddy.new" "$DIR/mode.caddy"
if reload; then
  echo "proxy mode: $MODE"
else
  mv "$DIR/mode.caddy.previous" "$DIR/mode.caddy"; reload || true
  echo "reload failed; previous mode restored"; exit 1
fi
