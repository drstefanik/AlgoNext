# Runbook Prod

## Step 1 — API locale

```bash
curl -i http://127.0.0.1:8000/health
```

## Step 2 — docker up

```bash
docker compose up -d --build
docker compose ps
docker compose logs -n 200 api
```

## Step 3 — Nginx

```bash
sudo nginx -t
sudo systemctl reload nginx
sudo tail -n 100 /var/log/nginx/error.log
```

## Step 4 — pubblico

```bash
curl -i https://api.nextgroupintl.com/health
```

## Interpretazione errori

- `connection refused` → API non ascolta / porta non esposta / container down.
- `timeout` → API bloccata / firewall / proxy loop.
- `502` → Nginx upstream down.
