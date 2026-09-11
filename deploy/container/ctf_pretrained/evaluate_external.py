"""Score a fine-tuned checkpoint on another dataset, and under degradation.

Two questions this answers, both of which the in-dataset number cannot:

  1. DOES IT GENERALISE? Fine-tuning on Celeb-DF reached held-out ROC-AUC
     0.9814, but on Celeb-DF identities and Celeb-DF's one manipulation family.
     FaceForensics++ uses four different methods (Deepfakes, Face2Face,
     FaceSwap, NeuralTextures). Nothing in the Celeb-DF result predicts those.
     The stock checkpoint already demonstrated how far this can fall: it scored
     0.9+ on its own training distribution and 0.4899 -- chance -- on Celeb-DF.

  2. DOES IT SURVIVE RE-ENCODING? Both evaluations so far used pristine dataset
     files. Real media arrives downscaled and recompressed. preprocess.py
     carries RandomDownscale and RandomJPEG for exactly this reason, but
     finetune_clips.py trains on clean crops, so robustness is untested.

    python evaluate_external.py --checkpoint reports/vit_finetuned \\
        --dataset ffpp --root /content/drive/MyDrive/ffpp_backup

Crops are extracted once and scored repeatedly under each condition, so the
sweep costs one pass of face detection rather than one per condition.

A large drop here is a finding, not a failure. Report it.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import data_sources as ds
from degrade import IMAGE_CONDITIONS, apply_image
from evaluate import metrics_report, print_report
from finetune_clips import extract_crops


def collect_external(dataset: str, root: Path, per_class: int, compression: str,
                     seed: int):
    """-> [(path, label, group)], capped and balanced per class."""
    records = ds.scan(dataset, root, compression=compression, verbose=True)
    by_label = {0: [], 1: []}
    for r in records:
        by_label[int(r.label)].append((Path(r.path), int(r.label), str(r.group_id)))
    for label, items in by_label.items():
        if not items:
            raise SystemExit(
                f"No {'fake' if label else 'real'} videos found under {root}. "
                f"Check the layout matches {dataset} (compression={compression}).")
        random.Random(seed).shuffle(items)
        del items[per_class:]
    jobs = by_label[0] + by_label[1]
    random.Random(seed).shuffle(jobs)
    return jobs


def score_under(model, processor, crops, labels, clips, device, batch_size,
                condition: str):
    """Clip-level P(fake) with `condition` applied to every crop first."""
    import torch
    from PIL import Image

    params = IMAGE_CONDITIONS[condition]
    probs = np.zeros(len(labels), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            sl = slice(start, start + batch_size)
            batch = [Image.fromarray(c) for c in crops[sl]]
            if params:
                batch = [apply_image(im, **params) for im in batch]
            x = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            # Index 1 is Deepfake, matching infer_pipeline._score().
            probs[sl] = torch.softmax(model(pixel_values=x).logits, 1)[:, 1].cpu().numpy()

    by_clip = {}
    for clip, p, y in zip(clips, probs, labels):
        by_clip.setdefault(clip, {"p": [], "y": int(y)})["p"].append(float(p))
    names = sorted(by_clip)
    y_true = np.array([by_clip[n]["y"] for n in names])
    y_prob = np.array([float(np.mean(by_clip[n]["p"])) for n in names])
    return y_true, y_prob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="a vit_finetuned directory written by finetune_clips.py")
    ap.add_argument("--dataset", default="ffpp", choices=["ffpp", "celebdf", "dfd"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--out-dir", default=Path("reports"), type=Path)
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--frames-per-video", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--conditions", nargs="*",
                    default=["clean", "jpeg_q50", "downscale_0.5",
                             "social_recompress", "heavy"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor

    unknown = [c for c in args.conditions if c not in IMAGE_CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; "
                         f"choose from {sorted(IMAGE_CONDITIONS)}")
    if not (args.checkpoint / "config.json").exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Point --checkpoint "
                         "at the vit_finetuned directory finetune_clips.py wrote.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = collect_external(args.dataset, args.root, args.per_class,
                            args.compression, args.seed)
    print(f"\n{len(jobs)} clips ({sum(1 for j in jobs if j[1] == 1)} fake, "
          f"{sum(1 for j in jobs if j[1] == 0)} real) from {args.dataset}")

    print("\nextracting crops:")
    crops, labels, clips = extract_crops(
        jobs, args.out_dir / f"{args.dataset}_cache", args.frames_per_video, device)
    if len(labels) == 0:
        raise SystemExit("No crops extracted -- no faces found in these videos.")
    print(f"{len(labels)} crops from {len(set(clips))} clips "
          f"({labels.mean():.1%} fake)")

    model = ViTForImageClassification.from_pretrained(str(args.checkpoint)).to(device)
    processor = ViTImageProcessor.from_pretrained(str(args.checkpoint))

    results = {}
    for cond in args.conditions:
        print(f"\n=== condition: {cond} {IMAGE_CONDITIONS[cond] or '(pristine)'}")
        y_true, y_prob = score_under(model, processor, crops, labels, clips,
                                     device, args.batch_size, cond)
        report = metrics_report(y_true, y_prob, threshold=0.5)
        print_report(f"{args.dataset} / {cond}", report)
        results[cond] = report

    print("\n" + "=" * 62)
    print(f"{'condition':<20}{'ROC-AUC':>10}{'accuracy':>11}{'baseline':>11}")
    for cond, r in results.items():
        print(f"{cond:<20}{r.get('roc_auc', float('nan')):>10.4f}"
              f"{r['accuracy']:>11.4f}{r.get('majority_baseline_accuracy', 0):>11.4f}")

    out = args.out_dir / f"external_{args.dataset}.json"
    out.write_text(json.dumps(
        {"dataset": args.dataset, "checkpoint": str(args.checkpoint),
         "compression": args.compression, "n_clips": len(set(clips)),
         "conditions": results}, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")

    clean_auc = results.get("clean", {}).get("roc_auc")
    if clean_auc is not None and clean_auc < 0.6:
        print(f"\nNOTE: ROC-AUC {clean_auc:.4f} on {args.dataset} against 0.9814 "
              "in-dataset. That gap is the cross-dataset generalisation failure "
              "this field is known for. It is a result worth reporting, not a "
              "bug to hide -- but do not claim general accuracy on the strength "
              "of the Celeb-DF number.")


if __name__ == "__main__":
    main()
