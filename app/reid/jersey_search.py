"""Bounded whole-window jersey search, independent of ReID candidate ranking."""

from collections import defaultdict
from typing import Any

import cv2

from app.reid.appearance import crop_from_normalized_bbox, evaluate_crop_quality
from app.reid.jersey_vision import JerseyCrop
from app.reid.team_color_guard import extract_kit_color_signature, signatures_compatible
from app.reid.window_logic import choose_descriptor_detections


def trim_kit_component(path, detections, seed_times, anchor_signature):
    """Stop propagation at the first frame that cannot attest the same kit.

    A tracker ID can jump to an opponent while keeping plausible motion. A
    sampled end-of-segment colour guard misses short switches. Check every
    observation before emitting a jersey-confirmed component instead.
    """
    if not detections or not seed_times:
        return []
    ordered = sorted(detections, key=lambda item: float(item["t"]))
    first = min(
        range(len(ordered)), key=lambda i: abs(float(ordered[i]["t"]) - min(seed_times))
    )
    last = min(
        range(len(ordered)), key=lambda i: abs(float(ordered[i]["t"]) - max(seed_times))
    )
    cap = cv2.VideoCapture(str(path))
    compatible = {}

    def matches(index):
        if index not in compatible:
            detection = ordered[index]
            cap.set(cv2.CAP_PROP_POS_MSEC, float(detection["t"]) * 1000)
            ok, frame = cap.read()
            crop = crop_from_normalized_bbox(frame, detection["bbox"]) if ok else None
            signature = extract_kit_color_signature(crop)
            compatible[index] = bool(
                signature and signatures_compatible(anchor_signature, signature) is True
            )
        return compatible[index]

    try:
        if not all(matches(i) for i in range(first, last + 1)):
            return []
        low, high = first, last
        while low > 0 and matches(low - 1):
            low -= 1
        while high + 1 < len(ordered) and matches(high + 1):
            high += 1
        return ordered[low : high + 1]
    finally:
        cap.release()


def scout_jerseys(
    verifier, path, track_map, window_start
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return sampling coordinates, never an identity association.

    Look outside the normal top-four appearance candidates. Every transmitted
    crop must pass the selected player's kit and image-quality checks first.
    The same response cannot supply two independent confirming reads.
    """
    summary = {"crops_considered": 0, "crops_read": 0, "matching_hints": 0}
    if not verifier.can_reacquire():
        return [], summary
    requests = defaultdict(list)
    for track_id, detections in track_map.items():
        if len(detections) < 2:
            continue
        for detection in choose_descriptor_detections(list(detections), 8):
            requests[float(detection["t"])].append((track_id, detection))
    cap = cv2.VideoCapture(str(path))
    usable = []
    try:
        for t in sorted(requests):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            for track_id, detection in requests[t]:
                crop = crop_from_normalized_bbox(frame, detection.get("bbox") or {})
                quality = evaluate_crop_quality(crop)
                if quality.width < 18 or quality.height < 36 or quality.sharpness < 40:
                    continue
                signature = extract_kit_color_signature(crop)
                if not signature or not signatures_compatible(
                    verifier.anchor_signature, signature
                ):
                    continue
                h = crop.shape[0]
                torso = crop[int(h * 0.12) : int(h * 0.70)]
                torso = cv2.resize(
                    torso, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
                )
                ok, encoded = cv2.imencode(
                    ".jpg", torso, [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                if not ok:
                    continue
                summary["crops_considered"] += 1
                # Original resolution matters: enlarging a tiny shirt does not
                # create legible digits. This ranking is only a sampling hint.
                priority = quality.score * min(2.0, quality.height / 80.0)
                usable.append(
                    (
                        priority,
                        track_id,
                        detection,
                        JerseyCrop(
                            encoded.tobytes(), window_start + t, quality.score, True
                        ),
                    )
                )
    finally:
        cap.release()
    # Round-robin across five-second intervals so one close-up cannot consume
    # the entire search. At most two distinct moments per local tracker ID.
    bins = defaultdict(list)
    for item in sorted(usable, key=lambda item: item[0], reverse=True):
        bins[int(float(item[2]["t"]) // 5)].append(item)
    chosen = []
    used = defaultdict(list)
    while bins and len(chosen) < 48:
        for key in sorted(list(bins)):
            pool = bins[key]
            while pool:
                item = pool.pop(0)
                track_id, t = item[1], float(item[2]["t"])
                if len(used[track_id]) >= 2 or any(
                    abs(t - other) < 0.6 for other in used[track_id]
                ):
                    continue
                chosen.append(item)
                used[track_id].append(t)
                break
            if not pool:
                del bins[key]
            if len(chosen) == 48:
                break
    readings = verifier.reader.read_many([item[3] for item in chosen])
    summary["crops_read"] = len(readings)
    hints = []
    for item, reading in zip(chosen, readings):
        if reading.get("legible") is True and reading.get("number") == verifier.target:
            hints.append({"t": float(item[2]["t"]), "bbox": item[2]["bbox"]})
    summary["matching_hints"] = len(hints)
    return hints, summary
