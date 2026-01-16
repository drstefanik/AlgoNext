# AlgoNext

## Backend environment variables (API + worker)

These variables are read from `.env` in local development (or from Docker/VPS envs). Set them **only** in the backend. The frontend should not build or rewrite MinIO URLs.

| Variable | Purpose |
| --- | --- |
| `MINIO_INTERNAL_ENDPOINT` | Used by API/worker for upload/download inside the server network. |
| `MINIO_PUBLIC_ENDPOINT` | Used **only** to create clickable links for the browser. |
| `MINIO_ACCESS_KEY` | MinIO access key for server-side S3 client. |
| `MINIO_SECRET_KEY` | MinIO secret key for server-side S3 client. |
| `MINIO_BUCKET` | Bucket name for assets. |

**VPS example values**

```bash
MINIO_INTERNAL_ENDPOINT=http://minio:9000
MINIO_PUBLIC_ENDPOINT=http://46.224.249.136:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=fnh
```

Notes:
- `MINIO_INTERNAL_ENDPOINT` and `MINIO_PUBLIC_ENDPOINT` must be different.
- Presigned URLs are generated with the internal endpoint and then rewritten to the public endpoint.
- If `MINIO_PUBLIC_ENDPOINT` is not set: development falls back to `http://localhost:9000`; production returns `"Asset link not configured"` instead of a broken link.
