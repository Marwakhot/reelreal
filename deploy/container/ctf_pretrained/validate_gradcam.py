"""Measure where the FINE-TUNED detector looks, and test the 1.5x naming cut.

STATUS.md carries a Grad-CAM table for six Celeb-DF crops in which every region
scores 0.09x-0.87x -- attention sits away from eyes, nose and mouth. That table
was produced with the off-the-shelf checkpoint, which `evaluate_clips.py`
measured at ROC-AUC 0.4899: chance. The attention of a model that discriminates
nothing explains nothing, so those numbers describe where a broken detector
looked and say nothing about the model actually serving traffic.

This reproduces the table on the fine-tuned weights, over many more clips, and
asks whether 1.5x is the right place to name a region.

WHAT CAN AND CANNOT BE VALIDATED HERE. There is no ground truth for "the model
ought to look at the mouth", so this does not score the cut against a correct
answer -- no such answer exists. Three things it can establish:

  1. THAT THE CUT CAN FIRE AT ALL. The rule it replaced required 40% of total
     CAM mass while the three discs cover ~41% of the crop between them, so it
     was unreachable by construction -- "never name a region" written to look
     like a threshold. A cut that fires on 0% of held-out clips has the same
     defect however principled its derivation, and that is checkable.

  2. WHERE 1.5x SITS IN THE OBSERVED DISTRIBUTION. Enrichment has a fixed null:
     attention spread evenly scores exactly 1.0 for every region whatever the
     disc sizes. So the measured spread of per-clip maxima says whether 1.5x is
     a modest step above the null or out in the tail where nothing lives.

  3. WHETHER FAKES AND REALS DIFFER. If the model attends to different regions
     on manipulated footage, the per-label distributions separate. If they do
     not, the row is a description of the model and nothing more -- which is
     exactly how the interface already words it.

It prints a recommendation derived from those numbers. It does not edit the cut:
`region_report(enrichment=...)` stays a stated constant, changed by hand once
someone has read the measurement.

Colab, with the checkpoint in Drive:

    python validate_gradcam.py --root /content/celeb-df \\
        --checkpoint /content/drive/MyDrive/vit_finetuned --out-dir reports

Or against named clips, to compare like for like with the six in STATUS.md:

    python validate_gradcam.py --clips id3_0006.mp4 id3_0007.mp4 --checkpoint ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REGIONS = ["eyes", "nose and cheeks", "mouth and jaw"]


def install_checkpoint(src: Path) -> Path:
    """Put the fine-tuned weights where VideoAnalyzer.load() looks for them.

    load() takes no checkpoint argument -- it looks for `vit_finetuned` beside
    infer_pipeline.py and falls back to the hub weights with a warning. Rather
    than reach around that, this copies the checkpoint into the one place the
    deployed loader reads, so what gets measured here is loaded by exactly the
    code path that serves traffic.
    """
    dest = HERE / "vit_finetuned"
    if src.resolve() == dest.resolve():
        return dest
    if not (src / "config.json").exists():
        raise SystemExit(
            f"No checkpoint at {src} -- expected a config.json written by "
            "finetune_clips.py's save_pretrained().")
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    print(f"installed checkpoint {src} -> {dest}")
    return dest


def held_out_clips(root: Path, per_class: int, holdout_frac: float, seed: int):
    """The same held-out identities finetune_clips.py evaluated on.

    finetune_clips.finetune_split() is the single definition of that split, so
    this asks it rather than rebuilding the collect + split_identities sequence
    and risking the two drifting apart. Measuring attention on clips the model
    trained on would describe memorisation, not detection.
    """
    from finetune_clips import finetune_split

    train_jobs, hold_jobs = finetune_split(root, per_class, holdout_frac, seed)
    print(f"{len(train_jobs) + len(hold_jobs)} clips -> {len(train_jobs)} train / "
          f"{len(hold_jobs)} held out "
          f"({len({j[2] for j in hold_jobs})} held-out identities)")
    return [(Path(p), int(label)) for p, label, _ in hold_jobs]


def collect_reports(analyzer, clips, n_frames):
    """-> list of per-clip records carrying the enrichment of every region."""
    rows, failures = [], 0
    for i, (path, label) in enumerate(clips, 1):
        try:
            result = analyzer.analyze(path, n_frames=n_frames, want_gradcam=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [{i}/{len(clips)}] {path.name}: FAILED {type(exc).__name__}: {exc}")
            failures += 1
            continue

        region = result.get("region") or {}
        enr = region.get("enrichment")
        if not enr:
            # No landmarks, no face, or Grad-CAM raised. Counted, not silently
            # dropped: a cut validated only on the clips that worked would be
            # validated on a filtered sample.
            print(f"  [{i}/{len(clips)}] {path.name}: no map "
                  f"({region.get('error') or region.get('reason') or 'no face'})")
            failures += 1
            continue

        row = {
            "clip": path.name,
            "label": label,
            "verdict": result.get("verdict"),
            "mean_prob": float(result.get("mean") or 0.0),
            "enrichment": {r: float(enr.get(r, 0.0)) for r in REGIONS},
            "shares": {r: float((region.get("shares") or {}).get(r, 0.0)) for r in REGIONS},
            "named_region": region.get("region"),
        }
        row["max_enrichment"] = max(row["enrichment"].values())
        row["argmax_region"] = max(row["enrichment"], key=row["enrichment"].get)
        rows.append(row)
        print(f"  [{i}/{len(clips)}] {path.name:<28} "
              + "  ".join(f"{r.split()[0]} {row['enrichment'][r]:.2f}x" for r in REGIONS)
              + f"   -> {row['named_region'] or 'spread out'}")
    return rows, failures


def sweep(rows, cuts):
    """Fraction of clips that would name a region at each candidate cut."""
    maxima = np.array([r["max_enrichment"] for r in rows], dtype=float)
    return {float(c): float((maxima >= c).mean()) for c in cuts}


def summarise(rows, cut: float) -> dict:
    maxima = np.array([r["max_enrichment"] for r in rows], dtype=float)
    labels = np.array([r["label"] for r in rows], dtype=int)

    per_region = {}
    for r in REGIONS:
        vals = np.array([row["enrichment"][r] for row in rows], dtype=float)
        per_region[r] = {
            "mean": float(vals.mean()), "median": float(np.median(vals)),
            "min": float(vals.min()), "max": float(vals.max()),
            "frac_above_1": float((vals > 1.0).mean()),
        }

    out = {
        "n_clips": len(rows),
        "cut": cut,
        "frac_naming_a_region": float((maxima >= cut).mean()),
        "max_enrichment": {
            "mean": float(maxima.mean()), "median": float(np.median(maxima)),
            "min": float(maxima.min()), "max": float(maxima.max()),
            "p90": float(np.percentile(maxima, 90)),
            "p99": float(np.percentile(maxima, 99)),
        },
        "per_region": per_region,
        "argmax_counts": {r: int(sum(1 for row in rows if row["argmax_region"] == r))
                          for r in REGIONS},
    }
    for name, sel in (("real", labels == 0), ("fake", labels == 1)):
        if sel.any():
            out[f"max_enrichment_{name}"] = {
                "n": int(sel.sum()), "mean": float(maxima[sel].mean()),
                "median": float(np.median(maxima[sel])),
                "frac_naming_a_region": float((maxima[sel] >= cut).mean()),
            }
    return out


def print_summary(s: dict, sweep_table: dict) -> None:
    print("\n" + "=" * 72)
    print(f"{s['n_clips']} clips with a usable attention map\n")

    print(f"{'region':<18}{'mean':>9}{'median':>9}{'min':>9}{'max':>9}{'>1.0x':>9}")
    for r, v in s["per_region"].items():
        print(f"{r:<18}{v['mean']:>9.3f}{v['median']:>9.3f}"
              f"{v['min']:>9.3f}{v['max']:>9.3f}{v['frac_above_1']:>8.0%}")

    m = s["max_enrichment"]
    print(f"\nper-clip maximum enrichment: mean {m['mean']:.3f}  "
          f"median {m['median']:.3f}  p90 {m['p90']:.3f}  max {m['max']:.3f}")
    print("(an evenly spread map scores exactly 1.000 for every region)")

    print("\nstrongest region, by clip count:")
    for r, n in s["argmax_counts"].items():
        print(f"  {r:<18} {n}")

    for name in ("real", "fake"):
        v = s.get(f"max_enrichment_{name}")
        if v:
            print(f"\n{name:<5} n={v['n']:<4} mean max-enrichment {v['mean']:.3f}  "
                  f"median {v['median']:.3f}  names a region {v['frac_naming_a_region']:.0%}")

    print("\ncut sweep -- share of clips that would name a region:")
    for c, frac in sweep_table.items():
        mark = "   <- current" if abs(c - s["cut"]) < 1e-9 else ""
        print(f"  {c:>5.2f}x   {frac:>6.1%}{mark}")


MIN_CLIPS_FOR_A_VERDICT = 30


def sample_is_too_thin(s: dict) -> str:
    """-> why this sample cannot settle the cut, or "" if it can.

    Checked BEFORE any verdict is composed, because the failure mode being
    guarded against is the one this project keeps finding in its own past work:
    a confident sentence derived from a handful of clips, quoted later as though
    it were the measurement. A run with no fakes in it is the sharper case --
    the naming branch exists to fire when a manipulated region draws the model's
    eye, so a set containing nothing manipulated has not tested it at all,
    however many real clips it holds.
    """
    n = s["n_clips"]
    has_fake = bool(s.get("max_enrichment_fake"))
    has_real = bool(s.get("max_enrichment_real"))
    reasons = []
    if not has_fake:
        reasons.append(
            "it contains no clips labelled fake, so the case the naming branch "
            "exists for -- attention concentrating on a manipulated region -- "
            "is untested here" if has_real else
            "the clips carry no labels (--clips mode), so fakes and reals "
            "cannot be told apart and the case the naming branch exists for "
            "is untested here")
    if n < MIN_CLIPS_FOR_A_VERDICT:
        reasons.append(f"{n} clips is below the {MIN_CLIPS_FOR_A_VERDICT} this "
                       "script treats as the minimum for a threshold decision")
    return "; ".join(reasons)


def recommend(s: dict, sweep_table: dict) -> str:
    """A verdict on the cut, phrased so it can be pasted into STATUS.md."""
    cut, fires = s["cut"], s["frac_naming_a_region"]
    p90 = s["max_enrichment"]["p90"]
    top = s["max_enrichment"]["max"]

    thin = sample_is_too_thin(s)
    if thin:
        return (
            f"NOT ENOUGH TO SETTLE THE CUT: {thin}. What this run does show, on "
            f"{s['n_clips']} clip(s): every region scored between "
            f"{min(v['min'] for v in s['per_region'].values()):.2f}x and "
            f"{max(v['max'] for v in s['per_region'].values()):.2f}x, and the "
            f"largest per-clip maximum was {top:.2f}x against the {cut:.2f}x cut. "
            "Record that as a measurement and leave the cut where it is. Re-run "
            "with --root over the held-out identities, which carries both labels "
            "at a usable sample size, before moving the threshold or touching the "
            "naming branch.")

    if top < cut:
        return (
            f"THE CUT NEVER FIRES. The largest per-clip enrichment across "
            f"{s['n_clips']} clips is {top:.2f}x, below the {cut:.2f}x cut, so no "
            f"clip can name a region -- the same defect the 40%-of-mass rule had, "
            f"in different arithmetic. Either lower the cut (p90 of the observed "
            f"maxima is {p90:.2f}x) or state plainly in the interface that this "
            f"model's attention never concentrates, and drop the naming branch "
            f"rather than leaving dead code that looks like a live threshold.")
    if fires < 0.02:
        return (
            f"THE CUT BARELY FIRES: {fires:.1%} of {s['n_clips']} clips. That is "
            f"defensible as a deliberately rare, high-bar statement, but say so "
            f"explicitly in STATUS.md -- a threshold that fires on one clip in "
            f"fifty is close enough to never firing that a reader should not have "
            f"to infer the difference. Observed p90 is {p90:.2f}x.")
    if fires > 0.5:
        return (
            f"THE CUT FIRES ON {fires:.0%} OF CLIPS, which makes naming a region "
            f"the normal case rather than a notable one. Enrichment above 1.0 is "
            f"expected whenever a face fills the crop, so a majority-firing cut "
            f"reports the geometry of face crops, not a property of the video. "
            f"Consider raising it towards p90 ({p90:.2f}x).")
    return (
        f"THE CUT IS SUPPORTED: {fires:.1%} of {s['n_clips']} clips name a region "
        f"at {cut:.2f}x, against a null of 1.0 and an observed p90 of {p90:.2f}x. "
        f"It fires often enough to be a live threshold and rarely enough that "
        f"naming a region still means something. Record the distribution in "
        f"STATUS.md alongside the number.")


def main() -> None:
    sys.path.insert(0, str(HERE))
    import finetune_clips

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path,
                     help="Celeb-DF root; the held-out identities are scored")
    src.add_argument("--clips", nargs="+", type=Path,
                     help="explicit video paths, for a like-for-like comparison "
                          "with a table you already have")
    ap.add_argument("--checkpoint", type=Path,
                    help="fine-tuned weights; copied next to infer_pipeline.py, "
                         "which is where VideoAnalyzer.load() looks")
    ap.add_argument("--out-dir", default=Path("reports"), type=Path)
    # Taken from finetune_clips.py rather than retyped, so the split reproduces
    # exactly and cannot drift if that script's defaults ever change.
    ap.add_argument("--per-class", type=int,
                    default=finetune_clips.FINETUNE_PER_CLASS)
    ap.add_argument("--holdout-frac", type=float,
                    default=finetune_clips.FINETUNE_HOLDOUT_FRAC)
    ap.add_argument("--seed", type=int, default=finetune_clips.FINETUNE_SEED)
    ap.add_argument("--limit", type=int, default=0,
                    help="score at most this many held-out clips (0 = all)")
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--cut", type=float, default=1.5,
                    help="the naming cut to test; must match region_report()'s "
                         "default, which this script never edits")
    ap.add_argument("--allow-hub-weights", action="store_true",
                    help="proceed on the off-the-shelf checkpoint anyway")
    args = ap.parse_args()

    if args.checkpoint:
        install_checkpoint(args.checkpoint)

    from infer_pipeline import VideoAnalyzer

    analyzer = VideoAnalyzer.load()

    # The whole point of re-running this is that the previous table came from a
    # checkpoint at chance. Producing a second table the same way would be a
    # wasted GPU hour and a misleading result, so it stops here by default.
    on_hub_weights = not (HERE / "vit_finetuned" / "config.json").exists()
    if on_hub_weights and not args.allow_hub_weights:
        raise SystemExit(
            "\nNo fine-tuned checkpoint -- VideoAnalyzer.load() fell back to the "
            "hub weights, which score ROC-AUC 0.4899 on held-out Celeb-DF clips. "
            "That is chance, and the attention of a model that discriminates "
            "nothing explains nothing: it is precisely the caveat this script "
            "exists to remove. Pass --checkpoint <dir>, or --allow-hub-weights "
            "if reproducing the old table is genuinely what you want.")

    if args.clips:
        clips = [(p, -1) for p in args.clips]
        missing = [p for p, _ in clips if not p.exists()]
        if missing:
            raise SystemExit(f"no such clip(s): {', '.join(map(str, missing))}")
        print(f"scoring {len(clips)} named clip(s); labels unknown, so the "
              "real/fake breakdown is skipped")
    else:
        clips = held_out_clips(args.root, args.per_class, args.holdout_frac, args.seed)
        if args.limit:
            clips = clips[:args.limit]
            print(f"limited to {len(clips)} clips")

    print(f"\nrunning Grad-CAM over {len(clips)} clip(s):")
    rows, failures = collect_reports(analyzer, clips, args.n_frames)
    if not rows:
        raise SystemExit("No clip produced a usable attention map.")

    cuts = [1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    if args.cut not in cuts:
        cuts = sorted(cuts + [args.cut])
    sweep_table = sweep(rows, cuts)
    summary = summarise(rows, args.cut)
    summary["n_failed"] = failures
    print_summary(summary, sweep_table)

    if failures:
        print(f"\n{failures} clip(s) produced no map and are excluded from every "
              "number above.")

    verdict = recommend(summary, sweep_table)
    print("\n" + "-" * 72)
    print(verdict)
    print("-" * 72)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "gradcam_validation.json"
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint) if args.checkpoint else "in-place",
        "on_hub_weights": on_hub_weights,
        "cut": args.cut,
        "summary": summary,
        "sweep": sweep_table,
        "recommendation": verdict,
        "clips": rows,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
