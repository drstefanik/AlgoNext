#!/usr/bin/env bash
set -euo pipefail

docker compose ps

curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/jobs

sudo tail -n 100 /var/log/nginx/algonext_error.log
