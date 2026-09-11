"""Is the model reading the face, or something else in the crop?

Three findings from this project point the same direction, and together they
are the classic signature of SHORTCUT LEARNING -- a model separating classes
using something real but incidental to the task, rather than the thing the
task is actually about:

  1. The off-the-shelf checkpoint scored 0.4899 (chance) on Celeb-DF; fine-tuning
     on Celeb-DF alone reached 0.9814 -- a very large in-dataset gain.
  2. That gain barely survives moving to a different manipulation family: FF++
     cross-dataset ROC-AUC is 0.7929 clean, 0.6301 under realistic recompression.
  3. Grad-CAM attention does not concentrate on the face at all, and does not
     even differ between real and fake clips (mean max-enrichment 0.517 fake vs
     0.499 real, on 120 held-out clips -- see STATUS.md).

Any one of these alone has an innocent explanation. Together, they are also
exactly what you would see if the model found something that correlates with
the Celeb-real / Celeb-synthesis label WITHOUT being the manipulation itself --
a compression or encoding difference between how the two folders' source
footage was produced, for instance. That would explain the in-dataset gain
(the shortcut is there), the cross-dataset collapse (the shortcut is
Celeb-DF-specific), and the diffuse attention (the model isn't looking at any
particular facial region because the signal isn't in one).

THE TEST: mask out the face and see if the model still discriminates.

  face_only        -- keep only the central ~77% of the crop (the original
                       detected face box, before preprocess.py's CROP_SCALE=1.3
                       expansion), mid-grey out the border.
  background_only   -- the complement: mid-grey out that central box, keep
                       only the border the expansion added.
  clean             -- unchanged, the reference point.

If the model's real signal lives in the face, background_only should collapse
towards chance and face_only should land close to clean. If background_only
holds up anywhere close to clean, that is direct evidence the model is reading
something outside the face -- and no amount of further training on this single
manipulation family will fix that; the fix is decontaminating what the training
data lets the model see, e.g. matching compression/encoding across the real and
fake sources, or training on more than one manipulation family so a
family-specific shortcut can't carry the whole signal.

    python diagnose_shortcut.py --root /content/celeb-df \\
        --checkpoint vit_finetuned --out-dir reports --limit 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config
from evaluate import metrics_report, print_report
from evaluate_external import auc_standard_error
from finetune_clips import extract_crops, finetune_split

# The original face box, before preprocess.py expands it by config.CROP_SCALE.
# CROP_SCALE=1.3 means the box you see here is 1/1.3 = ~77% of the crop's
# width and height, centered -- derived from the same constant the crop
# extraction pipeline uses, not a separately guessed number.
FACE_FRAC = 1.0 / config.CROP_SCALE

CONDITIONS = ["clean", "face_only", "background_only"]

# Flat mid-grey. Deliberately not each crop's own mean: a per-image mean leaks
# a little information about that image back into the "masked" region, which
# would muddy exactly the comparison this script exists to make.
FILL = 128


def masked(crop: np.ndarray, condition: str, face_frac: float = FACE_FRAC) -> np.ndarray:
    """-> crop with the face box or its complement painted flat grey."""
    if condition == "clean":
        return crop
    h, w = crop.shape[:2]
    bh, bw = int(round(h * face_frac)), int(round(w * face_frac))
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    out = crop.copy()
    if condition == "face_only":
        out[:] = FILL
        out[y0:y0 + bh, x0:x0 + bw] = crop[y0:y0 + bh, x0:x0 + bw]
    elif condition == "background_only":
        out[y0:y0 + bh, x0:x0 + bw] = FILL
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return out


def score_under(model, processor, crops, labels, clips, device, batch_size,
                condition: str):
    """Clip-level P(fake) with `condition` applied to every crop first.

    Mirrors evaluate_external.score_under, swapping the degradation transform
    for the masking one above -- same batching, same class-1-is-Deepfake
    convention, same clip-level mean aggregation aggregate.py uses in
    production, so this measures the same thing the deployed pipeline would.
    """
    import torch
    from PIL import Image

    probs = np.zeros(len(labels), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            sl = slice(start, start + batch_size)
            batch = [Image.fromarray(masked(c, condition)) for c in crops[sl]]
            x = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            probs[sl] = torch.softmax(model(pixel_values=x).logits, 1)[:, 1].cpu().numpy()

    by_clip = {}
    for clip, p, y in zip(clips, probs, labels):
        by_clip.setdefault(clip, {"p": [], "y": int(y)})["p"].append(float(p))
    names = sorted(by_clip)
    y_true = np.array([by_clip[n]["y"] for n in names])
    y_prob = np.array([float(np.mean(by_clip[n]["p"])) for n in names])
    return y_true, y_prob


def interpret(results: dict) -> str:
    clean = results.get("clean", {}).get("roc_auc")
    face = results.get("face_only", {}).get("roc_auc")
    bg = results.get("background_only", {}).get("roc_auc")
    bg_se = results.get("background_only", {}).get("roc_auc_se", float("nan"))
    if clean is None or face is None or bg is None:
        return "Could not compare -- one condition produced no usable metric."

    # Two standard errors above 0.5 is the same bar used elsewhere in this
    # project's reporting (see evaluate_external.py's sample-size note): below
    # it, "background_only beat chance" is not distinguishable from noise.
    bg_above_chance = (bg - 0.5) > 2 * bg_se if bg_se == bg_se else bg > 0.6

    if bg_above_chance:
        return (
            f"SHORTCUT SIGNAL FOUND: background_only ROC-AUC is {bg:.3f} "
            f"(+/-{bg_se:.3f}), meaningfully above chance with the face entirely "
            f"masked out. The model discriminates real from fake using content "
            f"outside the face box. clean={clean:.3f}, face_only={face:.3f} for "
            f"comparison. Training longer or harder on Celeb-DF alone will not "
            f"fix this -- it will most likely deepen reliance on whatever this "
            f"is. Next step: check whether real and fake source footage differ "
            f"systematically in compression, resolution or encoding, and either "
            f"match those before training or train on a second manipulation "
            f"family so no single shortcut can carry the whole signal.")
    if face < clean - 0.05:
        return (
            f"PARTIAL SIGNAL OUTSIDE THE FACE: background_only ({bg:.3f}) stays "
            f"near chance, but face_only ({face:.3f}) is meaningfully below "
            f"clean ({clean:.3f}). The border added by CROP_SCALE is "
            f"contributing something the masked-out version loses, even though "
            f"it cannot carry the signal alone. Worth a closer look, not yet a "
            f"confirmed shortcut.")
    return (
        f"NO SHORTCUT FOUND HERE: background_only ({bg:.3f}) sits at chance and "
        f"face_only ({face:.3f}) is close to clean ({clean:.3f}). The signal "
        f"this checkpoint uses lives inside the face box, which rules out a "
        f"shortcut living in the crop's border -- it does not rule out a "
        f"shortcut living inside the face itself (e.g. an artifact shared by "
        f"real and fake sources alike), which this test cannot see.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="Celeb-DF root")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(__file__).resolve().parent / "vit_finetuned")
    ap.add_argument("--out-dir", default=Path("reports"), type=Path)
    ap.add_argument("--per-class", type=int, default=300,
                    help="must match the fine-tune's --per-class or the "
                         "held-out identities reconstructed here will not be "
                         "the ones actually held out")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames-per-video", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap on held-out clips scored (0 = all)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor

    if not (args.checkpoint / "config.json").exists():
        raise SystemExit(
            f"No checkpoint at {args.checkpoint}. This test is meaningless on "
            "the off-the-shelf hub weights, which are at chance to begin with "
            "and have no real signal to localise.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, hold_jobs = finetune_split(args.root, args.per_class, args.holdout_frac,
                                  args.seed)
    if args.limit:
        hold_jobs = hold_jobs[:args.limit]
    print(f"{len(hold_jobs)} held-out clips "
          f"({sum(1 for j in hold_jobs if j[1] == 1)} fake, "
          f"{sum(1 for j in hold_jobs if j[1] == 0)} real)")

    print("\nextracting crops:")
    crops, labels, clips = extract_crops(
        hold_jobs, args.out_dir / "shortcut_cache", args.frames_per_video, device)
    if len(labels) == 0:
        raise SystemExit("No crops extracted -- no faces found in these videos.")
    print(f"{len(labels)} crops from {len(set(clips))} clips "
          f"({labels.mean():.1%} fake)")
    print(f"face_frac = {FACE_FRAC:.3f} (1 / config.CROP_SCALE = "
          f"1 / {config.CROP_SCALE})")

    model = ViTForImageClassification.from_pretrained(str(args.checkpoint)).to(device)
    processor = ViTImageProcessor.from_pretrained(str(args.checkpoint))

    results = {}
    for cond in CONDITIONS:
        print(f"\n=== condition: {cond}")
        y_true, y_prob = score_under(model, processor, crops, labels, clips,
                                     device, args.batch_size, cond)
        report = metrics_report(y_true, y_prob, threshold=0.5)
        n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
        report["roc_auc_se"] = auc_standard_error(
            report.get("roc_auc", float("nan")), n_pos, n_neg)
        print_report(f"shortcut / {cond}", report)
        results[cond] = report

    print("\n" + "=" * 62)
    print(f"{'condition':<20}{'ROC-AUC':>10}{'+/- SE':>9}{'accuracy':>11}")
    for cond, r in results.items():
        print(f"{cond:<20}{r.get('roc_auc', float('nan')):>10.4f}"
              f"{r.get('roc_auc_se', float('nan')):>9.4f}"
              f"{r['accuracy']:>11.4f}")

    verdict = interpret(results)
    print("\n" + "-" * 62)
    print(verdict)
    print("-" * 62)

    out = args.out_dir / "shortcut_diagnostic.json"
    out.write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "face_frac": FACE_FRAC,
         "n_clips": len(set(clips)), "conditions": results,
         "verdict": verdict}, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
