from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.benchmark.schema import (
    AnnotationFrame,
    BoundingBox,
    PredictionFrame,
    SequenceAnnotation,
    SequencePrediction,
)


@dataclass(frozen=True)
class GateThresholds:
    detection_f1_min: float = 0.75
    idf1_min: float = 0.65
    track_coverage_min: float = 0.60
    id_switches_per_100_max: float = 5.0
    hota_style_min: float = 0.55


@dataclass(frozen=True)
class _FrameMatch:
    gt_index: int
    prediction_index: int
    iou: float


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    first_x2 = first.x + first.w
    first_y2 = first.y + first.h
    second_x2 = second.x + second.w
    second_y2 = second.y + second.h

    intersection_x1 = max(first.x, second.x)
    intersection_y1 = max(first.y, second.y)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    union = first.w * first.h + second.w * second.h - intersection
    return intersection / union if union > 0 else 0.0


def _maximum_weight_assignment(
    weights: Sequence[Sequence[int]],
) -> list[tuple[int, int]]:
    """Return a deterministic maximum-weight one-to-one assignment.

    The implementation is the O(n^3) Hungarian algorithm. Rectangular matrices
    are padded with zero-weight dummy rows/columns. Zero-weight pairs are not
    returned.
    """

    row_count = len(weights)
    column_count = max((len(row) for row in weights), default=0)
    if row_count == 0 or column_count == 0:
        return []

    size = max(row_count, column_count)
    padded = [
        [
            int(weights[row][column])
            if row < row_count and column < len(weights[row])
            else 0
            for column in range(size)
        ]
        for row in range(size)
    ]
    maximum = max(max(row) for row in padded)
    costs = [[maximum - value for value in row] for row in padded]

    # Hungarian algorithm for minimum-cost assignment, 1-indexed.
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for row in range(1, size + 1):
        p[0] = row
        column0 = 0
        min_values = [10**30] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            current_row = p[column0]
            delta = 10**30
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = (
                    costs[current_row - 1][column - 1]
                    - u[current_row]
                    - v[column]
                )
                if current < min_values[column]:
                    min_values[column] = current
                    way[column] = column0
                if min_values[column] < delta:
                    delta = min_values[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = [-1] * size
    for column in range(1, size + 1):
        if p[column] != 0:
            assignment[p[column] - 1] = column - 1

    result: list[tuple[int, int]] = []
    for row, column in enumerate(assignment[:row_count]):
        if 0 <= column < column_count and padded[row][column] > 0:
            result.append((row, column))
    return result


def _match_frame(
    annotation: AnnotationFrame,
    prediction: PredictionFrame | None,
    iou_threshold: float,
) -> tuple[list[_FrameMatch], set[int], set[int], set[int]]:
    active_gt_indices = [
        index for index, obj in enumerate(annotation.objects) if not obj.ignore
    ]
    ignored_gt_indices = [
        index for index, obj in enumerate(annotation.objects) if obj.ignore
    ]
    tracks = prediction.tracks if prediction is not None else ()

    # Valid pairs receive a cardinality bonus large enough that one additional
    # match always dominates every possible total-IoU difference in the frame.
    # This creates a true lexicographic objective: match count first, IoU second.
    iou_scale = 100_000
    maximum_matches = min(len(active_gt_indices), len(tracks))
    cardinality_bonus = (maximum_matches + 1) * iou_scale
    weights: list[list[int]] = []
    ious: dict[tuple[int, int], float] = {}
    for gt_index in active_gt_indices:
        row: list[int] = []
        for prediction_index, track in enumerate(tracks):
            overlap = bbox_iou(annotation.objects[gt_index].bbox, track.bbox)
            ious[(gt_index, prediction_index)] = overlap
            if overlap >= iou_threshold:
                row.append(cardinality_bonus + int(round(overlap * iou_scale)))
            else:
                row.append(0)
        weights.append(row)

    matches: list[_FrameMatch] = []
    matched_gt_indices: set[int] = set()
    matched_prediction_indices: set[int] = set()
    for active_row, prediction_index in _maximum_weight_assignment(weights):
        gt_index = active_gt_indices[active_row]
        overlap = ious[(gt_index, prediction_index)]
        if overlap < iou_threshold:
            continue
        matches.append(
            _FrameMatch(
                gt_index=gt_index,
                prediction_index=prediction_index,
                iou=overlap,
            )
        )
        matched_gt_indices.add(gt_index)
        matched_prediction_indices.add(prediction_index)

    ignored_prediction_indices: set[int] = set()
    for prediction_index, track in enumerate(tracks):
        if prediction_index in matched_prediction_indices:
            continue
        if any(
            bbox_iou(annotation.objects[gt_index].bbox, track.bbox)
            >= iou_threshold
            for gt_index in ignored_gt_indices
        ):
            ignored_prediction_indices.add(prediction_index)

    return (
        matches,
        matched_gt_indices,
        matched_prediction_indices,
        ignored_prediction_indices,
    )


def _identity_assignment(
    pair_counts: Mapping[tuple[str, str], int],
) -> tuple[int, list[dict[str, Any]]]:
    identities = sorted({identity for identity, _ in pair_counts})
    track_ids = sorted({track_id for _, track_id in pair_counts})
    if not identities or not track_ids:
        return 0, []

    weights = [
        [pair_counts.get((identity, track_id), 0) for track_id in track_ids]
        for identity in identities
    ]
    assignment: list[dict[str, Any]] = []
    identity_true_positives = 0
    for identity_index, track_index in _maximum_weight_assignment(weights):
        matched_detections = weights[identity_index][track_index]
        if matched_detections <= 0:
            continue
        identity_true_positives += matched_detections
        assignment.append(
            {
                "identity": identities[identity_index],
                "track_id": track_ids[track_index],
                "matched_detections": matched_detections,
            }
        )
    assignment.sort(key=lambda item: (item["identity"], item["track_id"]))
    return identity_true_positives, assignment


def _identity_timeline_metrics(
    observations: Mapping[str, Sequence[tuple[int, str | None]]],
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    per_identity: list[dict[str, Any]] = []
    total_switches = 0
    total_fragmentations = 0
    mostly_tracked = 0
    mostly_lost = 0

    for identity in sorted(observations):
        timeline = sorted(observations[identity], key=lambda item: item[0])
        ground_truth_detections = len(timeline)
        matched_track_ids = [
            track_id for _, track_id in timeline if track_id is not None
        ]
        matched_detections = len(matched_track_ids)
        coverage = _safe_divide(matched_detections, ground_truth_detections)

        switches = 0
        fragmentations = 0
        last_matched_track: str | None = None
        gap_after_match = False
        current_gap = 0
        longest_gap = 0

        for _, track_id in timeline:
            if track_id is None:
                current_gap += 1
                longest_gap = max(longest_gap, current_gap)
                if last_matched_track is not None:
                    gap_after_match = True
                continue

            if last_matched_track is not None and track_id != last_matched_track:
                switches += 1
            if gap_after_match:
                fragmentations += 1
            last_matched_track = track_id
            gap_after_match = False
            current_gap = 0

        if coverage >= 0.80:
            mostly_tracked += 1
        if coverage <= 0.20:
            mostly_lost += 1

        total_switches += switches
        total_fragmentations += fragmentations
        per_identity.append(
            {
                "identity": identity,
                "ground_truth_detections": ground_truth_detections,
                "matched_detections": matched_detections,
                "coverage": round(coverage, 6),
                "distinct_track_ids": len(set(matched_track_ids)),
                "id_switches": switches,
                "fragmentations": fragmentations,
                "longest_gap_frames": longest_gap,
            }
        )

    return (
        per_identity,
        total_switches,
        total_fragmentations,
        mostly_tracked,
        mostly_lost,
    )


def _metrics_from_counts(counts: Mapping[str, float]) -> dict[str, float]:
    true_positives = float(counts.get("true_positives", 0))
    false_positives = float(counts.get("false_positives", 0))
    false_negatives = float(counts.get("false_negatives", 0))
    identity_true_positives = float(counts.get("identity_true_positives", 0))
    identity_false_positives = float(counts.get("identity_false_positives", 0))
    identity_false_negatives = float(counts.get("identity_false_negatives", 0))
    ground_truth_detections = float(counts.get("ground_truth_detections", 0))
    matched_iou_sum = float(counts.get("matched_iou_sum", 0))
    identities = float(counts.get("identities", 0))
    mostly_tracked = float(counts.get("mostly_tracked", 0))
    mostly_lost = float(counts.get("mostly_lost", 0))
    id_switches = float(counts.get("id_switches", 0))
    association_score_sum = float(counts.get("association_score_sum", 0))

    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    identity_precision = _safe_divide(
        identity_true_positives,
        identity_true_positives + identity_false_positives,
    )
    identity_recall = _safe_divide(
        identity_true_positives,
        identity_true_positives + identity_false_negatives,
    )
    detection_accuracy = _safe_divide(
        true_positives,
        true_positives + false_positives + false_negatives,
    )
    association_accuracy = _safe_divide(association_score_sum, true_positives)
    hota_style = math.sqrt(max(0.0, detection_accuracy * association_accuracy))

    return {
        "detection_precision": round(precision, 6),
        "detection_recall": round(recall, 6),
        "detection_f1": round(_f1(precision, recall), 6),
        "mean_matched_iou": round(
            _safe_divide(matched_iou_sum, true_positives), 6
        ),
        "detection_accuracy": round(detection_accuracy, 6),
        "association_accuracy": round(association_accuracy, 6),
        "hota_style_at_threshold": round(hota_style, 6),
        "track_coverage": round(
            _safe_divide(true_positives, ground_truth_detections), 6
        ),
        "identity_precision": round(identity_precision, 6),
        "identity_recall": round(identity_recall, 6),
        "idf1": round(_f1(identity_precision, identity_recall), 6),
        "id_switches_per_100_matches": round(
            _safe_divide(id_switches * 100.0, true_positives), 6
        ),
        "mostly_tracked_ratio": round(
            _safe_divide(mostly_tracked, identities), 6
        ),
        "mostly_lost_ratio": round(_safe_divide(mostly_lost, identities), 6),
    }


def evaluate_sequence(
    annotation: SequenceAnnotation,
    prediction: SequencePrediction,
    *,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    if annotation.video_id != prediction.video_id:
        raise ValueError(
            "video_id mismatch: "
            f"{annotation.video_id!r} != {prediction.video_id!r}"
        )
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")

    prediction_frames = {frame.frame_index: frame for frame in prediction.frames}
    pair_counts: Counter[tuple[str, str]] = Counter()
    prediction_track_counts: Counter[str] = Counter()
    identity_observations: dict[str, list[tuple[int, str | None]]] = defaultdict(list)

    ground_truth_detections = 0
    prediction_detections = 0
    ignored_predictions = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_iou_sum = 0.0

    for annotation_frame in annotation.frames:
        prediction_frame = prediction_frames.get(annotation_frame.frame_index)
        tracks = prediction_frame.tracks if prediction_frame else ()
        (
            matches,
            matched_gt_indices,
            matched_prediction_indices,
            ignored_prediction_indices,
        ) = _match_frame(annotation_frame, prediction_frame, iou_threshold)

        active_gt_indices = [
            index
            for index, obj in enumerate(annotation_frame.objects)
            if not obj.ignore
        ]
        ground_truth_detections += len(active_gt_indices)
        prediction_detections += len(tracks) - len(ignored_prediction_indices)
        for prediction_index, track in enumerate(tracks):
            if prediction_index not in ignored_prediction_indices:
                prediction_track_counts[track.track_id] += 1
        ignored_predictions += len(ignored_prediction_indices)
        true_positives += len(matches)
        false_negatives += len(active_gt_indices) - len(matched_gt_indices)
        false_positives += (
            len(tracks)
            - len(matched_prediction_indices)
            - len(ignored_prediction_indices)
        )
        matched_iou_sum += sum(match.iou for match in matches)

        match_by_gt = {
            match.gt_index: tracks[match.prediction_index].track_id
            for match in matches
        }
        for gt_index in active_gt_indices:
            identity = annotation_frame.objects[gt_index].identity
            track_id = match_by_gt.get(gt_index)
            identity_observations[identity].append(
                (annotation_frame.frame_index, track_id)
            )
            if track_id is not None:
                pair_counts[(identity, track_id)] += 1

    (
        per_identity,
        id_switches,
        fragmentations,
        mostly_tracked,
        mostly_lost,
    ) = _identity_timeline_metrics(identity_observations)
    identity_true_positives, identity_assignment = _identity_assignment(pair_counts)
    identity_false_negatives = ground_truth_detections - identity_true_positives
    identity_false_positives = prediction_detections - identity_true_positives
    identity_gt_counts = {
        identity: len(timeline)
        for identity, timeline in identity_observations.items()
    }
    association_score_sum = 0.0
    for (identity, track_id), matched_count in pair_counts.items():
        denominator = (
            identity_gt_counts.get(identity, 0)
            + prediction_track_counts.get(track_id, 0)
            - matched_count
        )
        association_jaccard = _safe_divide(matched_count, denominator)
        association_score_sum += matched_count * association_jaccard

    annotated_frame_indices = {frame.frame_index for frame in annotation.frames}
    unscored_prediction_frames = sum(
        1
        for frame in prediction.frames
        if frame.frame_index not in annotated_frame_indices
    )

    counts: dict[str, int | float] = {
        "annotation_frames": len(annotation.frames),
        "prediction_frames_scored": sum(
            1
            for frame in prediction.frames
            if frame.frame_index in annotated_frame_indices
        ),
        "prediction_frames_unscored": unscored_prediction_frames,
        "ground_truth_detections": ground_truth_detections,
        "prediction_detections": prediction_detections,
        "ignored_predictions": ignored_predictions,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matched_iou_sum": round(matched_iou_sum, 9),
        "association_score_sum": round(association_score_sum, 9),
        "identity_true_positives": identity_true_positives,
        "identity_false_positives": identity_false_positives,
        "identity_false_negatives": identity_false_negatives,
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "identities": len(identity_observations),
        "mostly_tracked": mostly_tracked,
        "mostly_lost": mostly_lost,
    }

    return {
        "video_id": annotation.video_id,
        "parameters": {"iou_threshold": iou_threshold},
        "counts": counts,
        "metrics": _metrics_from_counts(counts),
        "identity_assignment": identity_assignment,
        "per_identity": per_identity,
    }


def evaluate_dataset(
    pairs: Iterable[tuple[SequenceAnnotation, SequencePrediction]],
    *,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    sequences = [
        evaluate_sequence(annotation, prediction, iou_threshold=iou_threshold)
        for annotation, prediction in pairs
    ]
    if not sequences:
        raise ValueError("benchmark dataset is empty")

    aggregate_counts: defaultdict[str, float] = defaultdict(float)
    for sequence in sequences:
        for key, value in sequence["counts"].items():
            aggregate_counts[key] += float(value)

    integer_fields = {
        "annotation_frames",
        "prediction_frames_scored",
        "prediction_frames_unscored",
        "ground_truth_detections",
        "prediction_detections",
        "ignored_predictions",
        "true_positives",
        "false_positives",
        "false_negatives",
        "identity_true_positives",
        "identity_false_positives",
        "identity_false_negatives",
        "id_switches",
        "fragmentations",
        "identities",
        "mostly_tracked",
        "mostly_lost",
    }
    public_counts: dict[str, int | float] = {
        key: int(value) if key in integer_fields else round(value, 9)
        for key, value in aggregate_counts.items()
    }

    return {
        "schema_version": "tracking-benchmark-report-v1",
        "parameters": {"iou_threshold": iou_threshold},
        "sequence_count": len(sequences),
        "aggregate": {
            "counts": public_counts,
            "metrics": _metrics_from_counts(public_counts),
        },
        "sequences": sequences,
    }


def evaluate_quality_gate(
    report: Mapping[str, Any],
    thresholds: GateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or GateThresholds()
    metrics = (
        report["aggregate"]["metrics"]
        if "aggregate" in report
        else report["metrics"]
    )

    definitions = [
        ("detection_f1", ">=", thresholds.detection_f1_min),
        ("idf1", ">=", thresholds.idf1_min),
        ("track_coverage", ">=", thresholds.track_coverage_min),
        (
            "id_switches_per_100_matches",
            "<=",
            thresholds.id_switches_per_100_max,
        ),
        ("hota_style_at_threshold", ">=", thresholds.hota_style_min),
    ]
    checks: list[dict[str, Any]] = []
    for metric_name, comparator, threshold in definitions:
        actual = float(metrics.get(metric_name, 0.0))
        passed = actual >= threshold if comparator == ">=" else actual <= threshold
        checks.append(
            {
                "metric": metric_name,
                "actual": round(actual, 6),
                "comparator": comparator,
                "threshold": threshold,
                "passed": passed,
            }
        )

    return {
        "passed": all(check["passed"] for check in checks),
        "thresholds": asdict(thresholds),
        "checks": checks,
        "note": (
            "These are initial engineering gates for the tracking pipeline, "
            "not evidence that player evaluation is scientifically validated."
        ),
    }
