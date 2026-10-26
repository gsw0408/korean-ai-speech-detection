#!/bin/sh
# After a withdrawal or expiry, check that the donation can no longer be reached through the service or managed storage:
# quarantine, ledger bytes, SQLite side files, any same-content file on the host or donation filesystems (including Docker
# volumes and image layers), service logs, the journal and swap. Raw disk blocks are out of scope (no physical-erasure claim).
set -u
SHA=${1:?usage: deletion_audit.sh <sha256> <size-bytes> <file-id>}; SIZE=${2:?}; ID=${3:?}; DON=/srv/nii-donations; fail=0
pass() { echo "PASS $1"; }; failed() { echo "FAIL $1"; fail=1; }
[ ! -e "$DON/quarantine/$ID" ] && pass "file gone from quarantine" || failed "file still in quarantine"
[ -z "$(ls -A "$DON" | grep -E 'sqlite3-(journal|wal|shm)$')" ] && pass "no SQLite journal/WAL files" || failed "SQLite side files present"
python3 - "$DON/consent.sqlite3" "$SHA" "$ID" <<'PY' && pass "no trace of hash or id in ledger bytes" || failed "ledger bytes still contain hash or id"
import sys
raw = open(sys.argv[1], "rb").read()
sys.exit(1 if sys.argv[2].encode() in raw or sys.argv[3].encode() in raw else 0)
PY
copies=$(find / "$DON" -xdev -type f -size "${SIZE}c" -exec sha256sum {} + 2>/dev/null | grep -c "^$SHA ")
[ "$copies" -eq 0 ] && pass "no file with the same content on the host or donation filesystems" || failed "$copies file(s) with the same content"
legacy=$(docker volume inspect -f '{{.Mountpoint}}' release_donations 2>/dev/null || true)
[ -z "$legacy" ] || [ -z "$(find "$legacy" -type f -path '*quarantine*' 2>/dev/null)" ] && pass "legacy named volume holds no donation files" || failed "legacy named volume holds donation files"
[ "$(docker logs release-service-1 2>&1 | grep -c "$ID")" -eq 0 ] && [ "$(journalctl --no-pager 2>/dev/null | grep -c "$ID")" -eq 0 ] && pass "id absent from service logs and journal" || failed "id found in logs"
[ -z "$(swapon --noheadings 2>/dev/null)" ] && pass "no swap (memory copies are never paged to disk)" || failed "swap is enabled"
[ "$fail" -eq 0 ] && echo "DELETION AUDIT PASSED" || echo "DELETION AUDIT FAILED"
exit "$fail"
