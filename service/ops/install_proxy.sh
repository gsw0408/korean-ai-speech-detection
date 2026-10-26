#!/bin/sh
# Install or update the shared proxy directory /opt/nii/proxy from this ops/ folder (run as root on the host).
# The Caddyfile is a single-file bind mount, so it is rewritten in place; mode files live in the mounted directory.
set -eu
SRC=$(cd "$(dirname "$0")" && pwd); DIR=${NII_PROXY_DIR:-/opt/nii/proxy}
mkdir -p "$DIR"; chmod 755 "$DIR"
cat "$SRC/Caddyfile.public" > "$DIR/Caddyfile"
cp "$SRC/mode.normal.caddy" "$SRC/mode.emergency.caddy" "$DIR/"
[ -f "$DIR/mode.caddy" ] || cp "$SRC/mode.normal.caddy" "$DIR/mode.caddy"
chmod 644 "$DIR"/Caddyfile "$DIR"/mode*.caddy
echo "installed $DIR: $(sed -n 's/^vars nii_proxy_mode //p' "$DIR/mode.caddy") mode"
