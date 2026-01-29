#!/usr/bin/env bash

set +e

echo "=== CONTEXT ==="
hostname
date
whoami

echo "=== DOCKER ==="
if command -v docker >/dev/null 2>&1; then
  docker compose ps
  echo "--- docker compose logs (api) ---"
  docker compose logs -n 120 api
  echo "--- docker compose logs (nginx) ---"
  docker compose logs -n 120 nginx
else
  echo "docker missing"
fi

echo "=== LOCAL API ==="
curl -sS -i --max-time 5 http://127.0.0.1:8000/health || true
curl -sS -i --max-time 8 http://127.0.0.1:8000/jobs || true

echo "=== NGINX ==="
sudo nginx -t || true
sudo tail -n 120 /var/log/nginx/error.log || true

echo "=== PUBLIC ==="
curl -sS -i --max-time 10 https://api.nextgroupintl.com/health || true
