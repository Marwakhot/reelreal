"""Fine-tune the pretrained ViT on Celeb-DF face crops, then evaluate on clips.

WHY
---
evaluate_clips.py established that the off-the-shelf checkpoint
(prithivMLmods/Deep-Fake-Detector-v2-Model) scores at chance on Celeb-DF v2:
held-out ROC-AUC 0.49, and 0.47-0.53 for every clip statistic. The weights carry
no usable signal for this manipulation family, so no threshold, aggregate or
calibrator can recover one. Fine-tuning is the only remaining option.

    python finetune_clips.py --root /content/celeb-df --out-dir reports

WHAT IT DOES
  1. splits IDENTITIES into train / held-out before touching a single frame
  2. extracts face crops with the same preprocess.crops_from_frame used at
     inference, caching them so a re-run skips straight to training
  3. fine-tunes the ViT on crops from train identities only
  4. evaluates at CLIP level on held-out identities, the same way the interface
     aggregates, and writes accuracy, precision/recall, ROC-AUC and a confusion
     matrix

The split happens first, on identities, and nothing downstream can cross it.
A frame-level split would put crops of the same face in both halves and report
a model that has memorised people rather than learned manipulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import metrics_report, plot_confusion, print_report
from evaluate_clips import collect, split_by_identity


def split_identities(jobs, holdout_frac: float, seed: int):
    """Whole identities on one side or the other, both classes on each."""
    tr, ho = split_by_identity([label for _, label, _ in jobs],
                               [group for _, _, group in jobs],
                               holdout_frac, seed)
    return [jobs[i] for i in tr], [jobs[i] for i in ho]


def extract_crops(jobs, cache_dir: Path, frames_per_video: int, device: str):
    """-> (crops as uint8 HWC arrays, labels, clip ids). Cached as one .npz."""
    import video_io
    from preprocess import crops_from_frame, get_detector

    cache = cache_dir / f"crops_{frames_per_video}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        print(f"loaded {len(d['labels'])} cached crops from {cache}")
        return d["crops"], d["labels"], d["clips"]

    detector = get_detector(device)
    crops, labels, clips = [], [], []
    for i, (path, label, _group) in enumerate(jobs, 1):
        try:
            frames, _meta = video_io.sample_video(str(path), frames_per_video)
        except Exception as exc:
            print(f"  [{i}/{len(jobs)}] decode failed {path.name}: {exc}")
            continue
        kept = 0
        for f in frames:
            cs = crops_from_frame(f.image, detector)
            if not cs:
                continue
            crops.append(np.asarray(cs[0].image, dtype=np.uint8))
            labels.append(label)
            clips.append(str(path))
            kept += 1
        if i % 25 == 0 or i == len(jobs):
            print(f"  [{i}/{len(jobs)}] {len(crops)} crops so far")

    crops = np.stack(crops) if crops else np.zeros((0, 224, 224, 3), np.uint8)
    labels = np.array(labels, dtype=np.int64)
    clips = np.array(clips)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, crops=crops, labels=labels, clips=clips)
    print(f"cached {len(labels)} crops to {cache}")
    return crops, labels, clips


def _normalise(batch_uint8, processor, device):
    """uint8 HWC crops -> normalised NCHW tensor, using the model's own stats.

    The processor is asked for tensors directly rather than re-implementing its
    resize and normalisation, so training crops and inference crops go through
    identical arithmetic. A mismatch here is invisible and ruins the model.
    """
    import torch

    out = processor(images=list(batch_uint8), return_tensors="pt")
    return out["pixel_values"].to(device)


def train(crops, labels, processor, device, epochs, batch_size, lr):
    import torch
    from torch.optim import AdamW
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        "prithivMLmods/Deep-Fake-Detector-v2-Model").to(device)
    model.train()

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = torch.nn.CrossEntropyLoss()
    n = len(labels)
    order = np.arange(n)

    for ep in range(1, epochs + 1):
        np.random.shuffle(order)
        total, correct, running = 0, 0, 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            x = _normalise(crops[idx], processor, device)
            y = torch.tensor(labels[idx], device=device)

            logits = model(pixel_values=x).logits
            loss = lossf(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.detach()) * len(idx)
            correct += int((logits.argmax(1) == y).sum())
            total += len(idx)
            if (start // batch_size) % 20 == 0:
                print(f"  epoch {ep}  {total}/{n}  loss {running / total:.4f}  "
                      f"crop acc {correct / total:.4f}")
        print(f"epoch {ep} done: loss {running / total:.4f}  "
              f"crop accuracy {correct / total:.4f}")
    return model


def evaluate_clip_level(model, processor, crops, labels, clips, device, batch_size):
    """Per-crop P(fake) -> per-clip mean. Clip label is its crops' shared label."""
    import torch

    model.eval()
    probs = np.zeros(len(labels), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            sl = slice(start, start + batch_size)
            x = _normalise(crops[sl], processor, device)
            # Index 1 is Deepfake, per the checkpoint's id2label. Same
            # convention as infer_pipeline._score(); reading index 0 here
            # would invert every number while still looking plausible.
            probs[sl] = torch.softmax(model(pixel_values=x).logits, 1)[:, 1].cpu().numpy()

    by_clip = {}
    for clip, p, y in zip(clips, probs, labels):
        by_clip.setdefault(clip, {"probs": [], "label": int(y)})["probs"].append(float(p))
    names = sorted(by_clip)
    y_true = np.array([by_clip[c]["label"] for c in names])
    y_prob = np.array([float(np.mean(by_clip[c]["probs"])) for c in names])
    return y_true, y_prob, names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("reports"), type=Path)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--frames-per-video", type=int, default=6)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import ViTImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU. Fine-tuning on CPU will take hours.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    jobs = collect(args.root, args.per_class, args.seed)
    train_jobs, hold_jobs = split_identities(jobs, args.holdout_frac, args.seed)
    print(f"{len(jobs)} clips: {len(train_jobs)} train / {len(hold_jobs)} held out")
    print(f"identities: {len({j[2] for j in train_jobs})} train / "
          f"{len({j[2] for j in hold_jobs})} held out")
    overlap = {j[2] for j in train_jobs} & {j[2] for j in hold_jobs}
    assert not overlap, f"identity leak across the split: {overlap}"

    processor = ViTImageProcessor.from_pretrained(
        "prithivMLmods/Deep-Fake-Detector-v2-Model")

    print("\nextracting TRAIN crops:")
    xtr, ytr, _ = extract_crops(train_jobs, args.out_dir / "train_cache",
                                args.frames_per_video, device)
    print("\nextracting HELD-OUT crops:")
    xho, yho, cho = extract_crops(hold_jobs, args.out_dir / "holdout_cache",
                                  args.frames_per_video, device)
    if len(ytr) == 0 or len(yho) == 0:
        raise SystemExit("No crops extracted. Check that the videos contain faces.")
    print(f"\ntrain crops {len(ytr)} ({ytr.mean():.2%} fake)  "
          f"held-out crops {len(yho)} ({yho.mean():.2%} fake)")

    print("\nfine-tuning:")
    model = train(xtr, ytr, processor, device, args.epochs, args.batch_size, args.lr)

    print("\nevaluating at clip level on held-out identities:")
    y_true, y_prob, _names = evaluate_clip_level(
        model, processor, xho, yho, cho, device, args.batch_size)
    report = metrics_report(y_true, y_prob, threshold=0.5)
    print_report("HELD-OUT CLIPS (unseen identities), fine-tuned", report)
    plot_confusion(report, args.out_dir / "confusion_matrix_finetuned.png",
                   "Fine-tuned, held-out identities")

    ckpt = args.out_dir / "vit_finetuned"
    model.save_pretrained(ckpt)
    processor.save_pretrained(ckpt)
    (args.out_dir / "finetune_metrics.json").write_text(
        json.dumps({"holdout_metrics": report,
                    "n_train_clips": len(train_jobs),
                    "n_holdout_clips": len(hold_jobs),
                    "n_train_crops": int(len(ytr)),
                    "epochs": args.epochs, "lr": args.lr}, indent=2, default=float),
        encoding="utf-8")
    print(f"\nwrote {ckpt}")
    print(f"      {args.out_dir / 'finetune_metrics.json'}")
    print(f"      {args.out_dir / 'confusion_matrix_finetuned.png'}")


if __name__ == "__main__":
    main()
