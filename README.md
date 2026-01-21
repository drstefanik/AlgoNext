# AlgoNext

## Backend environment variables (API + worker)

These variables are read from `.env` in local development (or from Docker/VPS envs). Set them **only** in the backend. The frontend should not build or rewrite MinIO URLs.

Copy the example file and customize it:

```bash
cp .env.example .env
```

| Variable | Purpose |
| --- | --- |
| `S3_ENDPOINT_URL` | Used by API/worker for upload/download inside the server network. |
| `S3_PUBLIC_ENDPOINT_URL` | Used **only** to create clickable links for the browser. |
| `S3_ACCESS_KEY` | S3 access key for server-side client. |
| `S3_SECRET_KEY` | S3 secret key for server-side client. |
| `S3_BUCKET` | Bucket name for assets. |
| `S3_REGION` | Region name for S3. |
| `SIGNED_URL_EXPIRES_SECONDS` | Expiration for presigned URLs. |

**VPS example values**

```bash
S3_ENDPOINT_URL=http://minio:9000
S3_PUBLIC_ENDPOINT_URL=http://46.224.249.136:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=fnh
S3_REGION=us-east-1
SIGNED_URL_EXPIRES_SECONDS=3600
```

Notes:
- `S3_ENDPOINT_URL` and `S3_PUBLIC_ENDPOINT_URL` must be different.
- Presigned URLs for external clients are generated directly with `S3_PUBLIC_ENDPOINT_URL` (no rewrite).

Verification after deploy/restart:

```bash
JOB_ID="ce0314d8-5944-4bc6-b9df-60b9f1746e92"
SIGNED_URL="$(curl -s http://localhost:8000/jobs/$JOB_ID | jq -r '.data.preview_frames[0].signed_url')"
echo "$SIGNED_URL"
curl -s -o /dev/null -w "%{http_code}\n" "$SIGNED_URL"
```

Expected result: `200`.

## GitHub Actions Deploy

Create these GitHub Secrets for the repository:

- `VPS_HOST`: IP VPS (e.g. `46.224.249.136`)
- `VPS_USER`: utente SSH (e.g. `root` o `ubuntu`)
- `VPS_SSH_KEY`: private key completa (`-----BEGIN OPENSSH PRIVATE KEY-----...`)
- `VPS_SSH_PORT`: `22` (se non hai cambiato porta)

Recommended server setup (manual):

- Repository already cloned in `/opt/AlgoNext`.
- Docker and Docker Compose installed.
- `.env` present on the VPS (not committed to git).
