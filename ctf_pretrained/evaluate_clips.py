"""Held-out evaluation and clip calibration for the pretrained ViT pipeline.

WHY THIS EXISTS, SEPARATELY FROM fit_clip_calibration.py
--------------------------------------------------------
fit_clip_calibration.py is built around the *trained* pipeline: it wants a
model_best.pt checkpoint and the splits.json written by plan_splits.py. This
project ships no trained checkpoint -- VideoAnalyzer.load() pulls a pretrained
Hugging Face ViT and scores whole videos directly. This script is the same
procedure for that setup: point it at a folder of labelled videos, get a
held-out evaluation and a fitted clip calibrator.

    python evaluate_clips.py --root /content/celeb-df --out-dir reports

WHAT IT DOES
  1. scans Celeb-DF-v2 style folders for labelled clips
  2. runs the full video pipeline over each, caching per-frame probabilities so
     a re-run with different thresholds costs nothing
  3. splits BY IDENTITY, not by clip -- see _group_of()
  4. sweeps the per-frame flag threshold on the calibration split only
  5. fits the clip calibrator on the calibration split
  6. reports accuracy, precision/recall, ROC-AUC and a confusion matrix on the
     held-out split, which the calibrator never saw

The held-out numbers are the only ones fit to quote.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

import aggregate
from calibrate import (brier_score, expected_calibration_error,
                       fit_clip_calibrator, plot_reliability)
from evaluate import metrics_report, plot_confusion, print_report

# Celeb-DF-v2 layout. Real and fake live in separate folders; the label comes
# from the folder, never from parsing a filename.
FOLDERS = {
    "Celeb-real": 0,
    "YouTube-real": 0,
    "Celeb-synthesis": 1,
}


def _group_of(path: Path, label: int) -> str:
    """Identity group, so the same face cannot appear in both splits.

    Celeb-DF names real clips `id{N}_{clip}.mp4` and fakes
    `id{source}_id{target}_{clip}.mp4`. Grouping on the leading id keeps a
    person's real clips together with the fakes wearing their face. Without
    this the model can memorise an identity in calibration and be scored on
    the same face in the held-out split, which inflates every metric.

    YouTube-real filenames carry no id, so each is its own group.
    """
    stem = path.stem
    parts = stem.split("_")
    if parts and parts[0].startswith("id"):
        return parts[0]
    return stem


def collect(root: Path, per_class: int, seed: int):
    """-> list of (path, label, group), balanced and capped per class."""
    by_label = {0: [], 1: []}
    for folder, label in FOLDERS.items():
        d = root / folder
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.mp4")):
            by_label[label].append((p, label, _group_of(p, label)))

    for label, items in by_label.items():
        if not items:
            raise SystemExit(
                f"No videos found for label {label}. Expected Celeb-DF folders "
                f"{list(FOLDERS)} under {root}.")
        random.Random(seed).shuffle(items)
        del items[per_class:]

    jobs = by_label[0] + by_label[1]
    random.Random(seed).shuffle(jobs)
    return jobs


def score_clips(jobs, cache_path: Path, n_frames: int):
    """Run the pipeline over every clip, caching per-frame probabilities.

    Only the per-frame probabilities are cached. Every threshold, aggregate
    and verdict downstream is recomputed from them, so re-tuning never needs
    the videos again.
    """
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"loaded {len(cache)} cached clips from {cache_path}")

    analyzer = None
    records = []
    for i, (path, label, group) in enumerate(jobs, 1):
        key = str(path)
        if key not in cache:
            if analyzer is None:
                # Imported here, not at module scope, so a fully-cached re-run
                # needs neither torch nor the model download.
                from infer_pipeline import VideoAnalyzer
                analyzer = VideoAnalyzer.load()
            try:
                raw = analyzer.analyze(str(path), n_frames=n_frames,
                                       want_gradcam=False)
            except Exception as exc:
                print(f"  [{i}/{len(jobs)}] FAILED {path.name}: "
                      f"{type(exc).__name__}: {exc}")
                continue
            cache[key] = {"probs": raw.get("probs") or [],
                          "coverage": float(raw.get("coverage") or 0.0)}
            if i % 10 == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  [{i}/{len(jobs)}] {path.name}  "
                  f"{len(cache[key]['probs'])} frames scored")

        entry = cache[key]
        # A clip with no usable face carries no signal either way. Dropping it
        # here is the same refusal the interface makes, rather than scoring it
        # as though a verdict existed.
        if not entry["probs"]:
            continue
        records.append({"path": key, "label": label, "group": group,
                        "probs": entry["probs"], "coverage": entry["coverage"]})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return records


def group_split(records, holdout_frac: float, seed: int):
    """Split whole identity groups, never individual clips."""
    groups = sorted({r["group"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_hold = max(1, int(round(len(groups) * holdout_frac)))
    hold = set(groups[:n_hold])
    calib = [r for r in records if r["group"] not in hold]
    heldout = [r for r in records if r["group"] in hold]
    return calib, heldout


def _features(records, high_thresh):
    return np.array([aggregate.feature_vector(r["probs"], high_thresh)
                     for r in records], dtype=np.float64)


def _labels(records):
    return np.array([r["label"] for r in records], dtype=int)


def sweep_high_thresh(calib, grid):
    """Pick the per-frame flag threshold on the CALIBRATION split only.

    Chosen by ROC-AUC of the fitted calibrator's own output, evaluated on the
    calibration split. The held-out split is never consulted here -- that is
    the whole point of it.
    """
    from sklearn.metrics import roc_auc_score

    y = _labels(calib)
    best = (-1.0, None)
    for ht in grid:
        try:
            calib_model = fit_clip_calibrator(
                _features(calib, ht), y, aggregate.FEATURE_NAMES, ht)
        except ValueError:
            continue
        probs = np.array([aggregate.apply_clip_calibrator(r["probs"], calib_model)[0]
                          for r in calib])
        try:
            auc = roc_auc_score(y, probs)
        except ValueError:
            continue
        print(f"  high_thresh {ht:.2f} -> calibration AUC {auc:.4f}")
        if auc > best[0]:
            best = (auc, ht)
    if best[1] is None:
        raise SystemExit("Threshold sweep failed: no split had both classes.")
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path,
                    help="folder holding Celeb-real / YouTube-real / Celeb-synthesis")
    ap.add_argument("--out-dir", default=Path("reports"), type=Path)
    ap.add_argument("--per-class", type=int, default=150,
                    help="cap on clips per class (default 150)")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = collect(args.root, args.per_class, args.seed)
    print(f"{len(jobs)} clips selected "
          f"({sum(1 for j in jobs if j[1] == 1)} fake, "
          f"{sum(1 for j in jobs if j[1] == 0)} real)")

    records = score_clips(jobs, args.out_dir / "frame_probs_cache.json",
                          args.n_frames)
    print(f"{len(records)} clips produced usable frame scores")

    calib, heldout = group_split(records, args.holdout_frac, args.seed)
    print(f"\ncalibration: {len(calib)} clips over "
          f"{len({r['group'] for r in calib})} identities")
    print(f"held out:    {len(heldout)} clips over "
          f"{len({r['group'] for r in heldout})} identities")
    if not heldout or len(set(_labels(heldout))) < 2:
        raise SystemExit("Held-out split needs both classes. Use more clips or "
                         "a larger --holdout-frac.")

    print("\nsweeping per-frame threshold on the calibration split:")
    high_thresh = sweep_high_thresh(calib, [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)])
    print(f"chosen high_thresh: {high_thresh}")

    calibrator = fit_clip_calibrator(
        _features(calib, high_thresh), _labels(calib),
        aggregate.FEATURE_NAMES, high_thresh)

    y_hold = _labels(heldout)
    p_hold = np.array([aggregate.apply_clip_calibrator(r["probs"], calibrator)[0]
                       for r in heldout])

    # 0.5 on a calibrated probability, not a tuned operating point. Tuning the
    # decision threshold on the held-out split would contaminate the only
    # untouched data left.
    report = metrics_report(y_hold, p_hold, threshold=0.5)
    report["ece"] = expected_calibration_error(p_hold, y_hold)
    report["brier"] = brier_score(p_hold, y_hold)
    print_report("HELD-OUT (unseen identities)", report)

    plot_confusion(report, args.out_dir / "confusion_matrix.png",
                   "Held-out clips, unseen identities")
    plot_reliability(p_hold, y_hold, args.out_dir / "reliability.png",
                     "Clip-level calibration, held out")

    out = {
        "high_thresh": high_thresh,
        "decision_thresh": 0.5,
        "calibrator": calibrator,
        "n_calib_clips": len(calib),
        "n_holdout_clips": len(heldout),
        "n_calib_identities": len({r["group"] for r in calib}),
        "n_holdout_identities": len({r["group"] for r in heldout}),
        "holdout_metrics": report,
    }
    (args.out_dir / "clip_calibration.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out_dir / 'clip_calibration.json'}")
    print(f"      {args.out_dir / 'confusion_matrix.png'}")
    print(f"      {args.out_dir / 'reliability.png'}")

    if report.get("roc_auc", 0) < 0.5:
        print("\nWARNING: ROC-AUC below 0.5 means the ranking is INVERTED. "
              "Check that _score() reads the Deepfake class index (1), not 0.")


if __name__ == "__main__":
    main()
