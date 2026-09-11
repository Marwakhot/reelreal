"""Video -> verdict. Decode, detect, crop, score, aggregate, explain.

Rewritten to use the pre-trained Hugging Face Vision Transformer model
(prithivMLmods/Deep-Fake-Detector-v2-Model) for instant hackathon inference
without requiring local training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

import aggregate
import config
import evidence as ev
import video_io
from gradcam import GradCAM, overlay, region_report, vit_reshape, vit_target_layer
from model import get_device
from preprocess import crops_from_frame, get_detector


class VideoAnalyzer:
    def __init__(self, model, processor, ckpt: dict, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.arch = ckpt.get("arch", "vit")
        self.temperature = float(ckpt.get("temperature") or 1.0)
        self.clip_calibrator = ckpt.get("clip_calibrator")
        self.high_thresh = float(
            (self.clip_calibrator or {}).get("high_thresh")
            or ckpt.get("high_thresh") or config.DEFAULT_HIGH_THRESH)
        self.decision_thresh = float(
            ckpt.get("decision_thresh") or config.DEFAULT_DECISION_THRESH)
        self.detector = get_detector(device)

    @classmethod
    def load(cls, ckpt_path=None, device: str = None, strict_preproc: bool = True
             ) -> "VideoAnalyzer":
        """Load the detector: the fine-tuned checkpoint if present, else the hub one."""
        device = device or get_device()

        # A fine-tuned checkpoint beside this file wins. finetune_clips.py
        # writes it with save_pretrained(), so it loads through the same API as
        # the hub model and needs no separate code path.
        #
        # This matters more than a normal fallback. The stock hub weights were
        # measured at ROC-AUC 0.4899 on 300 held-out Celeb-DF clips -- chance.
        # Fine-tuning on Celeb-DF identities reached 0.9814 on 226 clips from 56
        # identities never seen in training. Running without the checkpoint is
        # running a detector that does not detect anything.
        finetuned = Path(__file__).resolve().parent / "vit_finetuned"
        if (finetuned / "config.json").exists():
            model_id, is_finetuned = str(finetuned), True
            print(f"Loading fine-tuned model from {finetuned}")
        else:
            model_id, is_finetuned = "prithivMLmods/Deep-Fake-Detector-v2-Model", False
            print(f"WARNING: no fine-tuned checkpoint at {finetuned}. Falling back "
                  f"to {model_id}, which scores at chance (ROC-AUC 0.49) on "
                  f"Celeb-DF. Verdicts will be meaningless.")

        model = ViTForImageClassification.from_pretrained(model_id).to(device)
        model.eval()

        processor = ViTImageProcessor.from_pretrained(model_id)

        if is_finetuned:
            # 0.5 on the clip mean is the operating point the held-out
            # evaluation actually reported: accuracy 0.9248, F1(fake) 0.9017,
            # ECE 0.0655. The fine-tuned outputs are calibrated enough that the
            # mean needs no further correction, so no clip calibrator is fitted.
            # 0.397 would buy recall 0.912 at a 5% false-positive rate if
            # missing fakes ever mattered more than accusing real footage.
            thresholds = {"decision_thresh": 0.50, "high_thresh": 0.50}
        else:
            # Provisional, swept on 12 clips and therefore in-sample. Retained
            # only so the fallback path is not entirely arbitrary; the model it
            # applies to is at chance regardless.
            thresholds = {"decision_thresh": 0.40, "high_thresh": 0.60}

        dummy_ckpt = {
            "arch": "vit",
            "temperature": 1.0,
            "clip_calibrator": None,
            **thresholds,
        }

        analyzer = cls(model, processor, dummy_ckpt, device)

        # A calibration file sitting beside the pipeline is picked up
        # automatically, so deploying a fitted calibrator is a file copy and
        # needs no code change. Absent one, the provisional thresholds above
        # stand and the interface says "Uncalibrated".
        default_calib = Path(__file__).resolve().parent / "clip_calibration.json"
        if analyzer.load_clip_calibration(default_calib):
            print(f"Loaded clip calibration from {default_calib}")

        return analyzer

    def load_clip_calibration(self, path) -> bool:
        """Load a calibrator written by evaluate_clips.py. True if adopted.

        Refuses a calibrator whose held-out ROC-AUC is below 0.5. That is not
        defensive padding: an earlier calibration run in this project scored
        0.357 because it was fitted while _score() read the wrong class index,
        and a below-chance calibrator does not merely underperform -- it
        confidently inverts every verdict while the interface reports itself
        as calibrated. Refusing to load it fails loudly instead.
        """
        p = Path(path)
        if not p.exists():
            return False
        try:
            report = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"Ignoring {p}: could not be read ({exc})")
            return False

        calibrator = report.get("calibrator")
        if not calibrator or not calibrator.get("coef"):
            print(f"Ignoring {p}: no calibrator coefficients in the file.")
            return False

        auc = (report.get("holdout_metrics") or {}).get("roc_auc")
        if auc is not None and auc < 0.5:
            print(f"REFUSING {p}: held-out ROC-AUC {auc:.3f} is below chance, "
                  "so this calibrator ranks clips backwards. Check that "
                  "_score() reads the Deepfake class index (1), then refit.")
            return False

        self.clip_calibrator = calibrator
        self.high_thresh = float(report.get("high_thresh")
                                 or calibrator.get("high_thresh")
                                 or self.high_thresh)
        self.decision_thresh = float(report.get("decision_thresh")
                                     or self.decision_thresh)
        return True

    @classmethod
    def untrained(cls, device: str = None) -> "VideoAnalyzer":
        """Fallback initializer."""
        return cls.load(device=device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _score(self, crop_images) -> np.ndarray:
        """Per-crop P(deepfake) from the Hugging Face Vision Transformer."""
        scores = []
        for im in crop_images:
            if isinstance(im, np.ndarray):
                im_pil = Image.fromarray(im)
            else:
                im_pil = im

            inputs = self.processor(images=im_pil, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits.float() / max(self.temperature, 1e-6)

            # The checkpoint declares id2label {0: 'Realism', 1: 'Deepfake'}, so
            # P(fake) is index 1. Reading index 0 here inverted every verdict.
            probs = torch.softmax(logits, dim=1)[0]
            scores.append(probs[1].item())
        return np.array(scores)

    def _explain(self, crop) -> Optional[Dict]:
        """Where the model looked on the most suspicious crop, named by landmark.

        Grad-CAM over the final transformer block, then region_report() turns
        the map into a region name only when one region draws at least 1.5x the
        attention an evenly spread map would put there -- its share of CAM mass
        over the share of image area its disc covers. Below that it returns
        region=None with the shares still filled in, so the interface can say
        "measured, nothing stood out" rather than naming whichever region
        happened to win or going silent as though nothing had been checked.

        This is a description of the model's attention, not evidence of
        manipulation. The report wording says so, and must keep saying so.

        Runs one extra forward-and-backward pass on a single crop, so it costs
        roughly one frame's inference. Any failure is caught and reported in
        `error` instead of losing the verdict that has already been computed --
        an explanation is a nicety, the verdict is the product.
        """
        try:
            inputs = self.processor(images=crop.image, return_tensors="pt")
            x = inputs["pixel_values"].to(self.device)
            side = crop.image.size[0] if hasattr(crop.image, "size")                 else crop.image.shape[0]

            with GradCAM(self.model, vit_target_layer(self.model),
                         reshape_transform=vit_reshape) as cam_fn:
                # class_idx 1 is Deepfake, the same index _score() reads. The
                # map answers "what drove the fake logit", so a different index
                # here would explain a different question than the one scored.
                cam = cam_fn(x, class_idx=1, out_size=side)

            # crop.landmarks are already in crop pixel coordinates and the
            # processor resizes the same square to the same 224, so the CAM and
            # the landmarks share one coordinate system with no rescaling.
            report = region_report(cam, crop.landmarks)
            if report is None:
                return {"region": None, "error": None,
                        "reason": "no usable face landmarks on this crop"}
            report.setdefault("error", None)
            return report
        except Exception as exc:
            return {"region": None, "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------
    def analyze(self, video_path, n_frames: int = None, want_gradcam: bool = True,
                policy: str = None) -> Dict:
        frames, meta = video_io.sample_video(video_path, n_frames)

        n_sampled = max(meta.n_requested, 1)

        frame_records: List[Dict] = []
        crops_by_frame = []
        for f in frames:
            crops = crops_from_frame(f.image, self.detector, policy=policy)
            if crops:
                crops_by_frame.append((f, crops))

        n_faces = len(crops_by_frame)
        coverage = n_faces / n_sampled

        flat, owner = [], []
        for i, (f, crops) in enumerate(crops_by_frame):
            for c in crops:
                flat.append(c)
                owner.append(i)

        probs_per_frame = []
        if flat:
            scores = self._score([c.image for c in flat])
            per_frame = {}
            for score, idx in zip(scores, owner):
                per_frame[idx] = max(per_frame.get(idx, -1.0), float(score))
            for i, (f, crops) in enumerate(crops_by_frame):
                p = per_frame.get(i, 0.0)
                probs_per_frame.append(p)
                frame_records.append({"index": int(f.index),
                                     "t_sec": round(float(f.t_sec), 3),
                                     "prob": round(float(p), 4),
                                     "n_faces": len(crops)})

        decision = aggregate.decide(
            probs_per_frame, coverage, calib=self.clip_calibrator,
            decision_thresh=self.decision_thresh, high_thresh=self.high_thresh)

        flagged_times = [r["t_sec"] for r, p in zip(frame_records, probs_per_frame)
                         if p > decision["high_thresh"]]

        region = None
        if want_gradcam and flat and probs_per_frame:
            best_i = int(np.argmax(probs_per_frame))
            best_crops = crops_by_frame[best_i][1]
            region = self._explain(max(best_crops, key=lambda c: c.face.area))

        result = {
            **decision,
            "video_path": str(video_path),
            "n_sampled": n_sampled,
            "n_decoded": meta.n_decoded,
            "n_faces": n_faces,
            "n_scored": len(probs_per_frame),
            "duration_sec": float(meta.duration_sec),
            "fps": float(meta.fps),
            "decode_mode": meta.decode_mode,
            "decode_errors": list(meta.errors),
            "min_coverage": config.MIN_FACE_COVERAGE,
            "temperature": self.temperature,
            "frames": frame_records,
            "probs": [round(float(p), 4) for p in probs_per_frame],
            "first_flagged_t": min(flagged_times) if flagged_times else None,
            "last_flagged_t": max(flagged_times) if flagged_times else None,
            "gradcam_t_sec": (frame_records[int(np.argmax(probs_per_frame))]["t_sec"]
                              if want_gradcam and probs_per_frame else None),
            "region": region,
        }
        result["headline"] = ev.headline(result["verdict"])
        result["guidance"] = ev.guidance_line(result["verdict"])
        result["evidence"] = ev.build_evidence(result)
        result["limitations"] = ev.LIMITATIONS
        return result


# --------------------------------------------------------------------------
# Convenience wrapper with a cached analyzer
# --------------------------------------------------------------------------
_analyzer: Optional[VideoAnalyzer] = None


def get_analyzer(ckpt_path=None, device: str = None) -> VideoAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = VideoAnalyzer.load(ckpt_path, device)
    return _analyzer


def analyze_video(video_path, ckpt_path=None, **kwargs) -> Dict:
    return get_analyzer(ckpt_path).analyze(video_path, **kwargs)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Analyse one video and print JSON.")
    ap.add_argument("video")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--no-gradcam", action="store_true")
    a = ap.parse_args()

    an = VideoAnalyzer.load(a.checkpoint)
    r = an.analyze(a.video, n_frames=a.n_frames, want_gradcam=not a.no_gradcam)
    r.pop("region", None)
    print(json.dumps(r, indent=2))