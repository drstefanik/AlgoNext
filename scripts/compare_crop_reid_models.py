#!/usr/bin/env python3
"""Offline identity-ranking diagnostic; never a production validation gate."""
from __future__ import annotations

import argparse
import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reid.appearance import _extract_hsv_descriptor
from app.reid.osnet_embedding import _load_osnet_source_module


class ReviewDataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.blocks = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "script" and attributes.get("id") == "review-data":
            if attributes.get("type") != "application/json" or self.active:
                raise ValueError("review-data must be one embedded JSON block")
            self.active = True
            self.parts = []

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.active:
            self.blocks.append("".join(self.parts))
            self.active = False


def decode_image(value, expected_hash):
    prefix = "data:image/jpeg;base64,"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError("Images must be embedded JPEGs")
    encoded = base64.b64decode(value[len(prefix):], validate=True)
    if hashlib.sha256(encoded).hexdigest() != expected_hash:
        raise ValueError("Image hash does not match the review manifest")
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable embedded JPEG")
    return image


def load_review(html_path, annotation_path):
    parser = ReviewDataParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    if len(parser.blocks) != 1 or parser.active:
        raise ValueError("Expected exactly one complete review-data JSON block")
    data = json.loads(parser.blocks[0])
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotation.get("schema_version") != "crop-identity-review-v1":
        raise ValueError("Unsupported annotation schema")
    for field in ("dataset_id", "video_id", "dataset_sha256"):
        if not data.get(field) or annotation.get(field) != data[field]:
            raise ValueError(f"Annotation {field} does not match the review file")
    if annotation.get("annotation_source") not in (
        "manual-review", "assistant-visual-review"
    ):
        raise ValueError("Annotation provenance must be explicit")
    if annotation.get("reference") != data["reference_metadata"]:
        raise ValueError("Annotation reference does not match the review file")
    manifest = [
        {k: v for k, v in row.items() if k not in ("crop_image", "scene_image")}
        for row in data["entries"]
    ]
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != data["dataset_sha256"]:
        raise ValueError("Review manifest hash mismatch")
    rows = data["entries"]
    by_file = {row["file"]: row for row in rows}
    annotations = annotation["entries"]
    if len(by_file) != len(rows) or len({r["file"] for r in annotations}) != len(annotations):
        raise ValueError("Duplicate crop identities")
    if {r["file"] for r in annotations} != set(by_file):
        raise ValueError("Annotations must list every crop, including unreviewed ones")
    selected = []
    counts = {"target": 0, "other": 0, "uncertain": 0, "unreviewed": 0}
    for row in annotations:
        original = by_file[row["file"]]
        for field, source in (
            ("crop_sha256", "crop_sha256"), ("time_sec", "t"),
            ("source_window", "source_window"), ("bbox", "bbox"),
        ):
            if row.get(field) != original[source]:
                raise ValueError(f"Crop provenance mismatch: {row['file']} / {field}")
        label = row.get("label")
        if label not in (None, "target", "other", "uncertain"):
            raise ValueError("Unknown identity label")
        counts[label or "unreviewed"] += 1
        if label in ("target", "other"):
            selected.append({**original, "label": label})
    if not counts["target"] or not counts["other"]:
        raise ValueError("At least one reviewed target and one reviewed other player are needed")
    reference = decode_image(data["reference"], data["reference_metadata"]["image_sha256"])
    images = [reference] + [decode_image(r["crop_image"], r["crop_sha256"]) for r in selected]
    return data, annotation, selected, counts, images


def load_model(constructor, path, torch):
    model = constructor(num_classes=1, pretrained=False, loss="softmax", use_gpu=False)
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    source = checkpoint.get("state_dict", checkpoint)
    features = {}
    for raw_key, value in source.items():
        key = raw_key.removeprefix("module.")
        if key.startswith("classifier."):
            continue
        if key in features:
            raise ValueError("Duplicate feature keys after prefix normalization")
        features[key] = value
    target = model.state_dict()
    expected = {k for k in target if not k.startswith("classifier.")}
    if set(features) != expected:
        raise ValueError("Checkpoint must contain every feature layer for this architecture")
    for key, value in features.items():
        if not isinstance(value, torch.Tensor) or value.shape != target[key].shape:
            raise ValueError(f"Incompatible feature tensor: {key}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite feature tensor: {key}")
    missing, unexpected = model.load_state_dict(features, strict=False)
    if set(missing) != {"classifier.weight", "classifier.bias"} or unexpected:
        raise ValueError("Feature checkpoint did not load completely")
    return model.eval().to("cpu")


def tensors_for(images, torch):
    mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(3, 1, 1)
    tensors = []
    for image in images:
        rgb = cv2.cvtColor(cv2.resize(image, (128, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        value = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255
        tensors.append(((value - mean) / std).unsqueeze(0))
    return tensors


def ranking_metrics(rows):
    positive = [r["hybrid_similarity"] for r in rows if r["label"] == "target"]
    negative = [r["hybrid_similarity"] for r in rows if r["label"] == "other"]
    if not positive or not negative:
        return None
    auc = sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))
    return {
        "pairwise_auc": auc,
        "target_crops": len(positive), "other_crops": len(negative),
        "target_mean": float(np.mean(positive)), "other_mean": float(np.mean(negative)),
        "minimum_target": min(positive), "maximum_other": max(negative),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-html", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--generic-weights", type=Path, required=True)
    parser.add_argument("--football-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 17))
    args = parser.parse_args()
    data, annotation, selected, counts, images = load_review(args.review_html, args.annotations)
    import torch

    torch.set_num_threads(args.threads)
    source = _load_osnet_source_module()
    tensors = tensors_for(images, torch)
    hsv = [np.array(_extract_hsv_descriptor(image).vector) for image in images]
    models = {}
    for name, constructor, path in (
        ("generic_osnet_x025", source.osnet_x0_25, args.generic_weights),
        ("soccernet_osnet_x10", source.osnet_x1_0, args.football_weights),
    ):
        model = load_model(constructor, path, torch)
        with torch.no_grad():
            model(tensors[0])  # Warm-up excluded from the measured inference time.
            start = time.perf_counter()
            vectors = []
            for tensor in tensors:
                vector = model(tensor).reshape(-1)
                norm = torch.linalg.vector_norm(vector)
                if not bool(torch.isfinite(norm)) or float(norm) <= 1e-12:
                    raise ValueError("Invalid model embedding")
                vectors.append((vector / norm).numpy())
            elapsed = time.perf_counter() - start
        scores = []
        for i, item in enumerate(selected, 1):
            learned = float(np.dot(vectors[0], vectors[i]))
            color = float(np.dot(hsv[0], hsv[i]))
            scores.append({
                "file": item["file"], "time_sec": item["t"],
                "source_window": item["source_window"], "label": item["label"],
                "learned_similarity": learned, "hybrid_similarity": .7 * learned + .3 * color,
            })
        models[name] = {
            "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "inference_seconds": elapsed, "inference_images": len(images),
            "ranking": ranking_metrics(scores),
            "per_window": {str(w): ranking_metrics([r for r in scores if r["source_window"] == w]) for w in sorted({r["source_window"] for r in scores})},
            "rows": scores,
        }
    report = {
        "schema_version": "crop-reid-diagnostic-v1", "validated": False,
        "production_eligible": False, "dataset_id": data["dataset_id"],
        "dataset_sha256": data["dataset_sha256"],
        "annotation_source": annotation["annotation_source"],
        "reviewer": annotation.get("reviewer"), "annotation_counts": counts,
        "reference": data["reference_metadata"], "cpu_threads": args.threads,
        "note": "Correlated crops from sampled windows and one anchor. AUC is a ranking diagnostic, not identity accuracy, full-match coverage, or player-ability validation. No production threshold is chosen.",
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"validated": False, "annotation_counts": counts, "models": {k: v["ranking"] for k, v in models.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
