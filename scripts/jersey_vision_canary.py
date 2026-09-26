"""Bounded production-key test on AlgoNext-owned cached frames. No source DB."""

import importlib.util
import json
import os
import sys

import cv2
import numpy as np

# A pre-deployment invocation may supply the reviewed module in /tmp.
if len(sys.argv) == 2:
    spec = importlib.util.spec_from_file_location("app.reid.jersey_vision", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

from app.reid.jersey_vision import JerseyCrop, JerseyReader
from app.reid.appearance import crop_from_normalized_bbox
from app.workers.tracking import _get_s3_client, S3_ENDPOINT_URL

os.environ["JERSEY_OCR_ENABLED"] = "1"
os.environ["JERSEY_OCR_MAX_CALLS"] = "4"
os.environ["JERSEY_OCR_MAX_SECONDS"] = "80"
job_id = "796f8c0f-94cd-4d2d-b8b1-a0f6ee5a5b60"
client = _get_s3_client(S3_ENDPOINT_URL)
asset = client.get_object(
    Bucket=os.environ["S3_BUCKET"], Key=f"jobs/{job_id}/frames/frame_0006.jpg"
)
try:
    raw = asset["Body"].read()
finally:
    asset["Body"].close()
frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
crop = crop_from_normalized_bbox(
    frame,
    {
        "x": 0.6835180759429932,
        "y": 0.5630893283420139,
        "w": 0.040279293060302736,
        "h": 0.10418120490180122,
    },
)
h = crop.shape[0]
torso = crop[int(h * 0.12) : int(h * 0.70)]
positive = cv2.resize(torso, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
blurred = cv2.resize(
    cv2.resize(torso, (3, 3), interpolation=cv2.INTER_AREA),
    (positive.shape[1], positive.shape[0]),
)
blank = np.full_like(positive, 225)
reader = JerseyReader()
results = []
for name, image, expected in [
    ("white_number_8", positive, 8),
    ("unreadable_blur", blurred, None),
    ("no_number", blank, None),
]:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise SystemExit("ENCODE_FAILED")
    result = reader.read(JerseyCrop(encoded.tobytes(), 1192.607, 1))
    results.append(
        {
            "case": name,
            "expected": expected,
            "observed": result.get("number"),
            "status": result["status"],
            "passed": result.get("number") == expected
            and result["status"] in {"READ", "UNREADABLE"},
        }
    )
print(
    json.dumps({"jersey_canary": results, "summary": reader.summary()}, sort_keys=True)
)
if not all(result["passed"] for result in results):
    raise SystemExit(1)
