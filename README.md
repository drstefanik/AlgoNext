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
- Presigned URLs are generated with the internal endpoint and then rewritten to the public endpoint.
- If `S3_PUBLIC_ENDPOINT_URL` is not set: development falls back to `http://localhost:9000`; production returns `"Asset link not configured"` instead of a broken link.
