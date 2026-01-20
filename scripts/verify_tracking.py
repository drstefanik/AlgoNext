import json
import os
import sys

import boto3
from botocore.client import Config


def get_s3_client(endpoint_url: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/verify_tracking.py <job_id> <tracking_key>")
        return 2

    job_id = sys.argv[1]
    tracking_key = sys.argv[2]
    endpoint = os.environ.get("S3_ENDPOINT_URL", "").strip()
    bucket = os.environ.get("S3_BUCKET", "").strip()
    if not endpoint or not bucket:
        print("Missing S3_ENDPOINT_URL or S3_BUCKET env vars")
        return 2

    s3 = get_s3_client(endpoint)
    obj = s3.get_object(Bucket=bucket, Key=tracking_key)
    data = json.loads(obj["Body"].read())

    method = data.get("method")
    coverage = data.get("coverage_pct", 0)
    bboxes = data.get("bboxes") or []
    print(
        f"job_id={job_id} method={method} coverage_pct={coverage} bboxes={len(bboxes)}"
    )
    if method != "yolo+bytetrack" or coverage <= 0 or len(bboxes) == 0:
        print("Tracking verification failed")
        return 1
    print("Tracking verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
