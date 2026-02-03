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
| `PREVIEW_FRAME_COUNT` | Number of preview frames to extract per job (default: 16). |
| `FULL_MATCH_MODE` | Enable full-match windowed tracking (set to `1` to use 45s windows with 10s overlap). |

**VPS example values**

```bash
S3_ENDPOINT_URL=http://minio:9000
S3_PUBLIC_ENDPOINT_URL=https://s3.nextgroupintl.com
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=fnh
S3_REGION=us-east-1
SIGNED_URL_EXPIRES_SECONDS=3600
```

Notes:
- `S3_ENDPOINT_URL` and `S3_PUBLIC_ENDPOINT_URL` must be different.
- Presigned URLs for external clients are generated directly with `S3_PUBLIC_ENDPOINT_URL` (no rewrite).
- MinIO must allow browser access to images via CORS. In Docker this is configured by `createbuckets`; otherwise apply it with `mc` (or the MinIO console):

```bash
cat << 'EOF' > cors.json
[
  {
    "AllowedOrigins": ["https://algonext-frontend.vercel.app"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag", "Content-Length", "Content-Type"],
    "MaxAgeSeconds": 3000
  }
]
EOF
mc cors set local/fnh cors.json

cat << 'EOF' > policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowFramesReadOnly",
      "Effect": "Allow",
      "Principal": {
        "AWS": ["*"]
      },
      "Action": ["s3:GetObject"],
      "Resource": [
        "arn:aws:s3:::fnh/jobs/*/frames/*",
        "arn:aws:s3:::fnh/jobs/*/candidates/*"
      ]
    }
  ]
}
EOF
mc anonymous set-json policy.json local/fnh
```

Verification after deploy/restart:

```bash
JOB_ID="ce0314d8-5944-4bc6-b9df-60b9f1746e92"
SIGNED_URL="$(curl -s http://localhost:8000/jobs/$JOB_ID | jq -r '.data.preview_frames[0].signed_url')"
echo "$SIGNED_URL"
curl -s -o /dev/null -w "%{http_code}\n" "$SIGNED_URL"
```

Expected result: `200`.

Frames debug (preview frames count):

```bash
curl -s "https://api.nextgroupintl.com/jobs/<id>/frames?count=16" | jq '.data.items | length'
```

Expected result: `16` (when at least 16 preview frames are available).

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
