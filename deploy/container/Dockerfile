# Container image for the REEL/REAL detection API. Built for Azure Container
# Apps; runs unmodified on Cloud Run, Container Instances and most other hosts.
#
# Python 3.11 because tokenizers 0.19.1 (pulled in by transformers 4.44.2) ships
# no wheels above 3.12, and scikit-learn 1.5.2 / matplotlib 3.9.2 stop at 3.13.
# On a newer Python pip would fall back to building those from source, which
# needs a Rust and a C++ toolchain and would fail here.
FROM python:3.11-slim

# libGL and libglib are needed by OpenCV; ffmpeg decodes the uploaded video.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. Everything the process writes to must be owned by that
# user, including the Hugging Face cache the model downloads into — the default
# cache path is root-owned and the download would fail.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# CPU-only torch. The default index would pull the CUDA build: ~2.5 GB of GPU
# libraries that a free CPU Space can never use, and large enough to fail the
# image build.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision \
         --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps facenet-pytorch==2.6.0

COPY --chown=user server/         ./server/
COPY --chown=user ctf_pretrained/ ./ctf_pretrained/
COPY --chown=user site/           ./site/

# app.py inserts PIPELINE_DIR into sys.path to import the detection pipeline.
ENV PIPELINE_DIR=/home/user/app/ctf_pretrained

# The fine-tuned checkpoint must be present, and must load. It is gitignored
# (~340 MB of safetensors), so a fresh clone will not have it -- copy it to
# deploy/container/ctf_pretrained/vit_finetuned/ before building.
#
# This is a hard failure rather than a fallback on purpose. The stock hub
# weights score at ROC-AUC 0.4899 on held-out Celeb-DF clips, which is chance,
# and a container serving them looks completely healthy: it returns verdicts,
# confidence numbers and evidence, all of them noise. Failing the build is the
# only point at which that is still cheap to notice.
RUN python -c "import sys; from pathlib import Path; \
d = Path('/home/user/app/ctf_pretrained/vit_finetuned'); \
sys.exit('MISSING %s -- see deploy/container/README.md' % d) if not (d / 'config.json').exists() else None; \
from transformers import ViTForImageClassification, ViTImageProcessor; \
ViTForImageClassification.from_pretrained(str(d)); ViTImageProcessor.from_pretrained(str(d)); \
print('fine-tuned checkpoint loads')"

# facenet-pytorch went in with --no-deps, so pip checked none of its imports.
# Build the face detector once here: a missing dependency then fails the build
# rather than the first upload, minutes into a live demo.
RUN python -c "import cv2; from facenet_pytorch import MTCNN; \
MTCNN(keep_all=True, post_process=False, device='cpu')"

# Ingress targets 7860. Container Apps is configured with --target-port 7860.
EXPOSE 7860
WORKDIR /home/user/app/server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
