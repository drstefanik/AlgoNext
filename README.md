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
| `PLAYER_REID_DESCRIPTOR_BACKEND` | `osnet_hybrid` uses the packaged OSNet person-ReID embedding plus kit colour; `hsv` is the rollback baseline. |
| `PLAYER_REID_OSNET_MODEL_PATH` | Local OSNet checkpoint path. The worker image prefetches the default MSMT17 x0.25 model. |
| `PLAYER_REID_LEARNED_FAIL_OPEN` | Fall back to the conservative HSV descriptor if OSNet cannot load. |
| `BALL_TRACKING_ENABLED` | Collect COCO sports-ball observations in the existing YOLO/ByteTrack pass. |
| `BALL_TRACKING_MIN_CONFIDENCE` | Minimum ball confidence retained for experimental trajectory/event diagnostics. |

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

## Match observability

The full-match tracker now emits a versioned `match-observability-v1` contract:

- robust multi-person median camera-motion compensation inside each window;
- ball observations from the same YOLO/ByteTrack pass, without rescanning the video;
- selected-player/ball proximity sequences as auditable event candidates;
- dynamic capability details (`available`, `experimental`, `foundation`, `unavailable`).

These signals are deliberately marked `validated: false` until their locked
real-match benchmarks pass. They enrich diagnostics but do not bypass the
player-evaluation truth gate.

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


## Async AI Scout Report API

The AI scout report can now be generated asynchronously via Celery:

- `POST /jobs/{id}/report` → enqueues `generate_report(job_id)`
- `GET /jobs/{id}/report` → returns report task status payload:

```json
{
  "ok": true,
  "data": {
    "status": "PENDING"
  }
}
```

When ready (`DONE`), the payload includes `report` with strict JSON fields:
`summary`, `strengths`, `risks`, `key_moments`, `training_plan_14_days`, `limitations`, `confidence`.

Environment:
- `OPENAI_MODEL` defaults to `gpt-5.2`
