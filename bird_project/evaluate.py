"""Reproducible bird detection evaluation and portable prediction caches.

AP uses TorchMetrics/COCO with configurable primary/supplementary detection caps
(defaults 100 and 300). The separate diagnostic IoU is configurable as well.
Thresholded diagnostics use score-ordered, class-aware, one-to-one IoU matching;
they are not AP. Ground-truth sizes are measured in original image pixels.
Reference: https://code.lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def box_iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(record, score_threshold=0.5, iou_threshold=0.5, max_detections=300):
    """Match all GT before any size slicing. Return original prediction/GT indices.

    Each prediction can claim at most one unmatched GT of the same class. Stable
    score ordering makes ties deterministic; a duplicate detection is an FP.
    """
    gt = record["ground_truth"]
    selected = sorted((i for i, score in enumerate(record["scores"]) if score >= score_threshold),
                      key=lambda i: -record["scores"][i])[:max_detections]
    unmatched = set(range(len(gt["boxes"])))
    matches, false_positives = [], []
    for pred_idx in selected:
        candidates = [(box_iou(record["boxes"][pred_idx], gt["boxes"][gt_idx]), gt_idx)
                      for gt_idx in sorted(unmatched) if record["labels"][pred_idx] == gt["labels"][gt_idx]]
        best = max(candidates, key=lambda item: item[0], default=(0.0, -1))
        if best[1] >= 0 and best[0] >= iou_threshold:
            matches.append({"prediction_index": pred_idx, "target_index": best[1], "iou": best[0]})
            unmatched.remove(best[1])
        else:
            false_positives.append(pred_idx)
    return {"matches": matches, "false_positive_indices": false_positives,
            "false_negative_indices": sorted(unmatched), "selected_prediction_indices": selected}


def size_bucket(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def density_bucket(n):
    return "0" if n == 0 else "1-5" if n <= 5 else "6-20" if n <= 20 else "21-50" if n <= 50 else "51+"


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def diagnostic_metrics(records, score_threshold=0.5, iou_threshold=0.5, max_detections=300):
    sizes = {key: {"ground_truth_count": 0, "matched_count": 0, "image_count": 0} for key in ("small", "medium", "large")}
    density = {key: {"image_count": 0, "ground_truth_count": 0, "tp": 0, "fp": 0, "fn": 0} for key in ("0", "1-5", "6-20", "21-50", "51+")}
    per_image = []
    total_tp = total_fp = total_fn = negative_count = negative_fp = negatives_with_fp = 0
    for record in records:
        matching = match_detections(record, score_threshold, iou_threshold, max_detections)
        tp, fp, fn = len(matching["matches"]), len(matching["false_positive_indices"]), len(matching["false_negative_indices"])
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn
        gt_boxes = record["ground_truth"]["boxes"]
        matched_gt = {item["target_index"] for item in matching["matches"]}
        present_sizes = set()
        for i, box in enumerate(gt_boxes):
            key = size_bucket(box)
            sizes[key]["ground_truth_count"] += 1
            sizes[key]["matched_count"] += int(i in matched_gt)
            present_sizes.add(key)
        for key in present_sizes:
            sizes[key]["image_count"] += 1
        group = density[density_bucket(len(gt_boxes))]
        for key, value in (("image_count", 1), ("ground_truth_count", len(gt_boxes)), ("tp", tp), ("fp", fp), ("fn", fn)):
            group[key] += value
        if not gt_boxes:
            negative_count += 1
            negative_fp += fp
            negatives_with_fp += int(fp > 0)
        per_image.append({"image_id": record["image_id"], "file_name": record["file_name"],
                          "ground_truth_count": len(gt_boxes), "tp": tp, "fp": fp, "fn": fn,
                          "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                          "f1": _ratio(2 * tp, 2 * tp + fp + fn)})
    for value in sizes.values():
        value["recall"] = _ratio(value["matched_count"], value["ground_truth_count"])
    for value in density.values():
        value["recall"] = _ratio(value["tp"], value["tp"] + value["fn"])
        value["precision"] = _ratio(value["tp"], value["tp"] + value["fp"])
    return {"score_threshold": score_threshold, "iou_threshold": iou_threshold, "max_detections": max_detections,
            "image_count": len(records), "ground_truth_count": total_tp + total_fn,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": _ratio(total_tp, total_tp + total_fp), "recall": _ratio(total_tp, total_tp + total_fn),
            "f1": _ratio(2 * total_tp, 2 * total_tp + total_fp + total_fn),
            "negative_images": {"image_count": negative_count, "false_positives": negative_fp,
                                "images_with_false_positives": negatives_with_fp,
                                "false_positives_per_image": _ratio(negative_fp, negative_count),
                                "fraction_with_false_positives": _ratio(negatives_with_fp, negative_count)},
            "size_recall": sizes, "density": density, "per_image": per_image,
            "size_definition": "original pixel area: small < 32^2; medium >= 32^2 and < 96^2; large >= 96^2",
            "negative_images_definition": "Images with zero ground-truth boxes in supplied labels. Empty labels do not prove the image contains no birds; FP means unmatched against those labels.",
            "matching": "All ground truth matched once in descending prediction score before size stratification; IoU >= threshold.",
            "undefined_values": "null denotes an empty denominator; these diagnostics are not AP."}


def validate_records(records):
    if not records:
        raise ValueError("Cannot evaluate an empty dataset.")
    ids, names = set(), set()
    for record in records:
        if record["image_id"] in ids or record["file_name"] in names:
            raise ValueError("Prediction cache contains duplicate image IDs or filenames.")
        ids.add(record["image_id"])
        names.add(record["file_name"])
        if Path(record["file_name"]).name != record["file_name"]:
            raise ValueError("file_name must be a filename, not a path.")
        if record["width"] <= 0 or record["height"] <= 0:
            raise ValueError("Image dimensions must be positive.")
        if not len(record["boxes"]) == len(record["scores"]) == len(record["labels"]):
            raise ValueError("Prediction boxes/scores/labels lengths differ.")
        gt = record["ground_truth"]
        if len(gt["boxes"]) != len(gt["labels"]):
            raise ValueError("Ground-truth boxes and labels lengths differ.")
        for label in record["labels"] + gt["labels"]:
            if label != 1:
                raise ValueError("This single-class evaluator requires bird=1 labels.")
        for box in record["boxes"] + gt["boxes"]:
            if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Invalid xyxy box in {record['file_name']}: {box}")
            if box[0] < -0.01 or box[1] < -0.01 or box[2] > record["width"] + 0.01 or box[3] > record["height"] + 0.01:
                raise ValueError(f"Out-of-image box in {record['file_name']}: {box}")
        if not all(math.isfinite(s) and 0 <= s <= 1 for s in record["scores"]):
            raise ValueError("Prediction scores must be finite probabilities.")


def canonical_boxes(boxes):
    """Use the float32 coordinates consumed by COCO and source A's cached targets.

    This normalizes serialization only; original annotation files are unchanged.
    It also preserves real coordinate differences larger than float32 precision.
    """
    return [[struct.unpack("f", struct.pack("f", value))[0] for value in box] for box in boxes]


def annotation_fingerprint(records):
    """Identify filenames/dimensions/float32 GT, not image pixel contents."""
    canonical = [{"file_name": record["file_name"], "width": record["width"], "height": record["height"],
                  "ground_truth": {"boxes": canonical_boxes(record["ground_truth"]["boxes"]),
                                   "labels": record["ground_truth"]["labels"]}}
                 for record in sorted(records, key=lambda r: r["file_name"])]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def calculate_metrics(records, max_detections=300, score_threshold=0.5, *,
                      primary_max_detections=100, diagnostic_iou_threshold=0.5):
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    validate_records(records)
    if primary_max_detections <= 10 or max_detections < primary_max_detections:
        raise ValueError("COCO caps must satisfy 10 < primary_max_detections <= max_detections.")
    if not 0 <= score_threshold <= 1 or not 0 < diagnostic_iou_threshold <= 1:
        raise ValueError("Diagnostic score must be in [0,1] and IoU in (0,1].")
    predictions, targets = [], []
    for record in records:
        predictions.append({"boxes": torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4),
                            "scores": torch.tensor(record["scores"], dtype=torch.float32),
                            "labels": torch.tensor(record["labels"], dtype=torch.int64)})
        targets.append({"boxes": torch.tensor(record["ground_truth"]["boxes"], dtype=torch.float32).reshape(-1, 4),
                        "labels": torch.tensor(record["ground_truth"]["labels"], dtype=torch.int64)})
    scores_by_cap = {}
    for cap in sorted({primary_max_detections, max_detections}):
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", max_detection_thresholds=[1, 10, cap],
                                      extended_summary=(cap != 100))
        # The model may legitimately return >100 boxes; cap is handled by COCO.
        metric.warn_on_many_detections = False
        metric.update(predictions, targets)
        values = metric.compute()
        scalar_values = {}
        for name, value in values.items():
            if name in ("classes", "ious", "precision", "recall", "scores") or name.endswith("_per_class"):
                continue
            if value.numel() == 1:
                scalar = float(value.item())
                scalar_values[name] = scalar if math.isfinite(scalar) and scalar >= 0 else None
        if cap != 100:
            # pycocotools' summarize() hardcodes maxDets=100 for its first AP
            # entry, even when configured with [1,10,300]. Average the same
            # accumulated COCO precision tensor directly: T,R,K,area=all,M=last.
            precision = values["precision"][:, :, :, 0, -1]
            valid = precision[precision > -1]
            scalar_values["map"] = float(valid.mean().item()) if valid.numel() else None
        scores_by_cap[str(cap)] = scalar_values
    return {"schema_version": 1, "image_count": len(records), "annotation_fingerprint": annotation_fingerprint(records),
            "map": scores_by_cap[str(primary_max_detections)]["map"],
            "map_50": scores_by_cap[str(primary_max_detections)]["map_50"],
            "primary_max_detections": primary_max_detections, "metrics_by_max_detections": scores_by_cap,
            "diagnostics": diagnostic_metrics(records, score_threshold=score_threshold,
                                               iou_threshold=diagnostic_iou_threshold, max_detections=max_detections),
            "protocol": {"ap_iou_thresholds": "0.50:0.05:0.95", "ap_recall_thresholds": "0.00:0.01:1.00",
                         "labels": {"1": "bird"}, "ap_score_filter": "all cached model detections; no additional threshold",
                         "undefined_ap": "null means COCO reports no evaluable targets for this slice",
                         "custom_cap_map": "For non-100 caps, AP50:95 is the mean of valid COCO accumulated precision[T,R,K,area=all,maxDets=cap]; avoids pycocotools summarize() hardcoded 100.",
                         "annotation_fingerprint": "SHA-256 of ordered filenames, dimensions and float32-normalized ground truth; excludes image pixels"}}
