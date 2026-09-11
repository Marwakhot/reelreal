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


def drop_finetune_identities(jobs, root, per_class: int, holdout_frac: float,
                             seed: int):
    """-> jobs with every identity the fine-tune trained on removed.

    Calibration turns a model score into a probability, and the mapping is only
    correct for the score distribution it was fitted on. Scores on faces the
    model memorised during fine-tuning sit further from the decision boundary
    and are more confident than anything an unseen face produces, so a
    calibrator fitted on them is mis-specified for the only case that matters
    in deployment. The held-out metrics are inflated too, but that is the lesser
    problem: a wrong number can be re-measured, whereas this ships a file that
    the loader accepts, reports is_calibrated true for, and prints a confident
    band from.

    Filtering is by IDENTITY rather than by clip. A different clip of a face the
    model trained on is still that face, and Celeb-DF's whole point is that the
    same person appears many times.
    """
    import finetune_clips

    train_ids = finetune_clips.finetune_train_identities(
        root, per_class, holdout_frac, seed)
    kept = [j for j in jobs if j[2] not in train_ids]

    dropped = len(jobs) - len(kept)
    dropped_ids = {j[2] for j in jobs} & train_ids
    print(f"\nexcluding {len(train_ids)} identities the fine-tune trained on "
          f"(per-class {per_class}, holdout {holdout_frac}, seed {seed})")
    print(f"  dropped {dropped} of {len(jobs)} clips "
          f"across {len(dropped_ids)} identities")
    print(f"  kept    {len(kept)} clips "
          f"({sum(1 for j in kept if j[1] == 1)} fake, "
          f"{sum(1 for j in kept if j[1] == 0)} real) "
          f"over {len({j[2] for j in kept})} unseen identities")

    if not kept:
        raise SystemExit(
            "Every clip belongs to an identity the fine-tune trained on, so "
            "there is nothing left to calibrate against. Raise --per-class "
            "above --finetune-per-class, or check that the --finetune-* "
            "arguments match the run that produced the checkpoint.")
    if len(set(j[1] for j in kept)) < 2:
        raise SystemExit(
            "Only one class survives the identity filter. A calibrator needs "
            "both, and the metrics would be meaningless. Raise --per-class.")
    return kept


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


def split_by_identity(labels, groups, holdout_frac: float, seed: int):
    """-> (train_idx, holdout_idx). Whole identities, stratified by label.

    Grouping alone is not enough. Assigning identities to sides at random can
    hand one side a single class -- a smoke test on a small sample produced a
    training split that was 0% fake, which teaches the model to answer "real"
    to everything and looks like a plausible run until the metrics arrive.
    StratifiedGroupKFold keeps identities whole AND both classes present.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(labels)
    n_splits = max(2, min(int(round(1.0 / max(holdout_frac, 1e-6))), len(set(groups))))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, hold_idx = next(sgkf.split(np.zeros(len(y)), y, groups))

    for name, idx in (("train/calibration", train_idx), ("held-out", hold_idx)):
        if len(set(y[idx])) < 2:
            raise SystemExit(
                f"The {name} split contains only one class. Use more clips, or "
                f"a --holdout-frac closer to 0.5.")
    assert not (set(np.asarray(groups)[train_idx]) &
                set(np.asarray(groups)[hold_idx])), "identity leak across split"
    return train_idx, hold_idx


def group_split(records, holdout_frac: float, seed: int):
    """Split whole identity groups, never individual clips."""
    tr, ho = split_by_identity([r["label"] for r in records],
                               [r["group"] for r in records], holdout_frac, seed)
    return [records[i] for i in tr], [records[i] for i in ho]


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
    # Imported here, not at module scope: finetune_clips imports collect and
    # split_by_identity from this module, so a top-level import back would be a
    # cycle. Only the argparse defaults and drop_finetune_identities need it.
    import finetune_clips

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path,
                    help="folder holding Celeb-real / YouTube-real / Celeb-synthesis")
    ap.add_argument("--out-dir", default=Path("reports"), type=Path)
    ap.add_argument("--per-class", type=int, default=150,
                    help="cap on clips per class (default 150)")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--n-frames", type=int, default=None,
                    help="frames sampled per clip; leave unset to use "
                         "config.N_SAMPLE_FRAMES, which is what the deployed "
                         "pipeline uses. Lowering it to save time refits the "
                         "calibrator to a frame count production never sees.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-finetune-identities", action="store_true",
                    help="drop every clip whose identity the fine-tune trained "
                         "on. Required for an honest calibrator whenever the "
                         "checkpoint being scored was fine-tuned on this root.")
    ap.add_argument("--finetune-per-class", type=int,
                    default=finetune_clips.FINETUNE_PER_CLASS,
                    help="--per-class the fine-tune used, to reproduce its split")
    ap.add_argument("--finetune-holdout-frac", type=float,
                    default=finetune_clips.FINETUNE_HOLDOUT_FRAC,
                    help="--holdout-frac the fine-tune used")
    ap.add_argument("--finetune-seed", type=int,
                    default=finetune_clips.FINETUNE_SEED,
                    help="--seed the fine-tune used")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = collect(args.root, args.per_class, args.seed)
    print(f"{len(jobs)} clips selected "
          f"({sum(1 for j in jobs if j[1] == 1)} fake, "
          f"{sum(1 for j in jobs if j[1] == 0)} real)")

    if args.exclude_finetune_identities:
        jobs = drop_finetune_identities(
            jobs, args.root, args.finetune_per_class,
            args.finetune_holdout_frac, args.finetune_seed)
    else:
        # Not a warning that can be silenced by ignoring it: the resulting file
        # deploys with is_calibrated true and a confident band, and nothing
        # downstream can tell it was fitted on memorised faces.
        print("\nNOTE: --exclude-finetune-identities was not passed. If the "
              "checkpoint being scored was fine-tuned on clips under this "
              "root, the calibrator is being fitted on faces it memorised and "
              "the held-out metrics below are inflated.")

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
        # Recorded in the file itself, because this is the one fact that cannot
        # be recovered from a deployed calibrator by looking at it. Two files
        # with identical coefficients and similar metrics mean entirely
        # different things depending on this flag.
        "excluded_finetune_identities": bool(args.exclude_finetune_identities),
        "n_frames": args.n_frames,
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
