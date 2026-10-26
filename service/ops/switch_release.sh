#!/bin/sh
# Switch the public stack to /opt/nii/release-<tag> (run as root on the host).
# While any donation is stored, a target is refused unless its image has withdrawal and expiry and its compose
# overlay uses the donation filesystem and the shared proxy; after switching, withdrawal must answer 404, not 403.
set -eu
TAG=${1:?usage: switch_release.sh <tag> [--allow-audio-storage]}
ALLOW_AUDIO=false
case ${2:-} in
  "") [ "$#" -eq 1 ] || exit 2;;
  --allow-audio-storage) [ "$#" -eq 2 ] || exit 2; ALLOW_AUDIO=true;;
  *) echo "REFUSED: unknown argument"; exit 2;;
esac
NEW=/opt/nii/release-$TAG; OLD=$(cat /opt/nii/CURRENT_RELEASE); DON=/srv/nii-donations
compose() { dir=$1; shift; docker compose -p release --env-file "$dir/.env" -f "$dir/docker-compose.prod.yml" -f "$dir/docker-compose.public.yml" "$@"; }
[ -f "$NEW/.env" ] && [ -f "$NEW/docker-compose.public.yml" ] || { echo "REFUSED: $NEW is not a staged release"; exit 2; }

# This applies even with an empty ledger: old images can otherwise resume audio storage after a rollback.
image=$(sed -n 's/^IMAGE=//p' "$NEW/.env")
audio_problems=""
docker run --rm --network none --entrypoint python "$image" -c "import sys; from app.settings import Settings; sys.exit(0 if 'donation_store_audio' in Settings.__dataclass_fields__ else 1)" >/dev/null 2>&1 \
  || audio_problems="$audio_problems image-without-audio-storage-setting"
python3 - "$NEW/.env" <<'PY' || audio_problems="$audio_problems env-not-exactly-no-audio"
import sys
lines = [line for line in open(sys.argv[1]).read().splitlines() if line.strip().split('=', 1)[0].removeprefix('export ').strip() == 'DONATION_STORE_AUDIO']
sys.exit(0 if lines == ['DONATION_STORE_AUDIO=false'] else 1)
PY
compose "$NEW" config --format json | python3 -c '
import json, sys
value = json.load(sys.stdin)["services"]["service"].get("environment", {}).get("DONATION_STORE_AUDIO")
sys.exit(0 if value == "false" else 1)' || audio_problems="$audio_problems compose-not-no-audio"
if [ -n "$audio_problems" ]; then
  if [ "$ALLOW_AUDIO" != true ]; then
    echo "REFUSED: audio storage is not disabled:$audio_problems"; exit 6
  fi
  echo "WARNING: AUDIO STORAGE OVERRIDE -- WRITTEN OWNER APPROVAL REQUIRED:$audio_problems" >&2
fi

stored=$(python3 - "$DON" <<'PY'
import os, sqlite3, sys
quarantine, ledger = os.path.join(sys.argv[1], "quarantine"), os.path.join(sys.argv[1], "consent.sqlite3")
files = [name for name in os.listdir(quarantine) if not name.startswith(".tmp-")] if os.path.isdir(quarantine) else []
rows = sqlite3.connect(ledger).execute("SELECT COUNT(*) FROM donations").fetchone()[0] if os.path.exists(ledger) else 0
print(len(files) + rows)
PY
)
problems=""
docker run --rm --network none --entrypoint python "$image" -c "import sys; from donations.store import DonationStore as D; sys.exit(0 if all(hasattr(D, n) for n in ('withdraw', 'expire', 'reconcile', 'purge_all')) else 1)" >/dev/null 2>&1 \
  || problems="$problems image-without-withdrawal-or-expiry"
compose "$NEW" config --format json | python3 -c '
import json, sys
services = json.load(sys.stdin)["services"]
mounts = lambda name: {(v.get("source"), v.get("target")) for v in services[name].get("volumes", [])}
ok = ("/srv/nii-donations", "/data/donations") in mounts("service") and {("/opt/nii/proxy", "/etc/nii-proxy"), ("/opt/nii/proxy/Caddyfile", "/etc/caddy/Caddyfile")} <= mounts("proxy")
sys.exit(0 if ok else 1)' || problems="$problems overlay-without-donation-mount-or-shared-proxy"
if [ -n "$problems" ]; then
  if [ "$stored" -gt 0 ]; then
    echo "REFUSED: $stored stored donation file(s)/record(s) and target problems:$problems"
    echo "Use emergency mode or 'DONATIONS_ENABLED=false' in the current release instead."; exit 3
  fi
  echo "WARNING: no stored donations, allowing target with problems:$problems"
fi

proxy_before=$(docker inspect -f '{{.State.StartedAt}}' release-proxy-1 2>/dev/null || echo none)
restore() { echo "restoring $OLD"; cd "$OLD"; compose "$OLD" up -d --wait --wait-timeout 300 || true; }
cd "$NEW"
if ! compose "$NEW" up -d --wait --wait-timeout 300; then echo "FAILED: target did not become healthy"; restore; exit 4; fi
site=$(sed -n 's/^SITE_ADDRESS=//p' "$NEW/.env")
code=$(curl -sk -o /tmp/nii-switch-probe.json -w '%{http_code}' -X DELETE --resolve "$site:443:127.0.0.1" "https://$site/api/donations/switch-probe-$(date +%s)" || true)
if [ -z "$problems" ] && { [ "$code" != 404 ] || ! grep -q withdrawal_not_found /tmp/nii-switch-probe.json; }; then
  echo "FAILED: withdrawal probe answered $code"; restore; exit 5
fi
echo "$NEW" > /opt/nii/CURRENT_RELEASE
proxy_after=$(docker inspect -f '{{.State.StartedAt}}' release-proxy-1 2>/dev/null || echo none)
echo "SWITCHED $OLD -> $NEW (withdrawal probe $code; proxy $( [ "$proxy_before" = "$proxy_after" ] && echo 'kept running' || echo 'was recreated'))"
