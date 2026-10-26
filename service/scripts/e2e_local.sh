#!/bin/sh
set -eu
BASE_URL=${BASE_URL:-http://127.0.0.1:8000} python scripts/e2e_local.py
