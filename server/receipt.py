"""Signed verdict receipts.

WHAT A RECEIPT IS
-----------------
A small JSON object, signed with an Ed25519 key held by this server, saying:

    "this detector, at this version, produced this verdict for a file whose
     SHA-256 is <hash>, at this time."

Anyone holding the public key can verify that statement offline, and anyone
holding the video file can confirm the receipt refers to *that* file by hashing
it themselves. Neither step requires trusting the person who sent it.

WHAT A RECEIPT IS NOT
---------------------
It is **not** a claim that the video is genuine, and not a claim that the
verdict is correct. It binds a result to a file and to this detector. A signed
receipt saying "synthetic, 0.91" is exactly as trustworthy as the detector
itself — the signature only removes the possibility that someone edited the
number afterwards or attributed it to the wrong clip.

This distinction is the whole point of the feature, and it is why the receipt
carries no field that could be read as a verdict on authenticity beyond what
the detector already said.

WHY IT IS WORTH HAVING
----------------------
A screenshot of a detector's output is unfalsifiable in the wrong direction:
anyone can edit the percentage in an image editor, or pair a real report with a
different video. A receipt makes both of those detectable.

THE KEY
-------
Set REELREAL_SIGNING_KEY to a base64 32-byte Ed25519 seed:

    python server/receipt.py keygen

With no key set, the server generates a throwaway one at startup so the
feature works out of the box in development. That key dies with the process,
so receipts signed by it stop verifying after a restart. /v1/public-key
reports `ephemeral: true` in that case, and the verify page says so on screen —
an ephemeral key must never be presented as a durable attestation.

CANONICAL FORM
--------------
Signatures are taken over JSON serialised with sorted keys and no whitespace.
Any two implementations that follow that rule produce identical bytes, which is
what lets a browser verify a receipt a Python process signed.

One extra rule, and the reason for it: **every non-integer number in the
payload is carried as a decimal string**, never as a JSON float. Python writes
the float 0.0 as `0.0` and JavaScript writes it as `0`, so a receipt with a
confidence of exactly 0.0 or 1.0 would serialise to different bytes in the two
languages and fail to verify in the browser while verifying fine in Python.
Strings remove the ambiguity completely rather than relying on both runtimes
choosing the same shortest representation.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

RECEIPT_VERSION = 1
ALGORITHM = "Ed25519"

_private_key: Optional[Ed25519PrivateKey] = None
_is_ephemeral = False


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    # Accept both standard and URL-safe alphabets, and tolerate missing padding,
    # because receipts get pasted through chat apps and URLs before they arrive.
    text = text.strip().replace("-", "+").replace("_", "/")
    return base64.b64decode(text + "=" * (-len(text) % 4))


def canonical_bytes(payload: Dict[str, Any]) -> bytes:
    """The exact bytes that get signed and verified.

    sort_keys makes key order irrelevant; the tight separators remove every
    byte of optional whitespace. Both sides must agree on this or no signature
    will ever check out.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------
def _load_key() -> Tuple[Ed25519PrivateKey, bool]:
    """Return (private key, is_ephemeral), loading it once per process."""
    global _private_key, _is_ephemeral
    if _private_key is not None:
        return _private_key, _is_ephemeral

    seed = os.environ.get("REELREAL_SIGNING_KEY", "").strip()
    if seed:
        raw = _unb64(seed)
        if len(raw) != 32:
            raise ValueError(
                "REELREAL_SIGNING_KEY must decode to exactly 32 bytes, got %d. "
                "Generate one with: python server/receipt.py keygen" % len(raw))
        _private_key = Ed25519PrivateKey.from_private_bytes(raw)
        _is_ephemeral = False
    else:
        _private_key = Ed25519PrivateKey.generate()
        _is_ephemeral = True
        print("[receipt] No REELREAL_SIGNING_KEY set - signing with a throwaway "
              "key that dies with this process. Receipts will stop verifying "
              "after a restart. Generate a real key with: "
              "python server/receipt.py keygen", file=sys.stderr)

    return _private_key, _is_ephemeral


def public_key_bytes() -> bytes:
    key, _ = _load_key()
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def key_id(public_raw: Optional[bytes] = None) -> str:
    """Short fingerprint of a public key, so a receipt can name the key that signed it.

    Truncated to 16 hex characters: long enough that two keys in this project
    will not collide, short enough to read out loud. It is an identifier, not a
    security boundary — verification uses the full key.
    """
    raw = public_raw if public_raw is not None else public_key_bytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def public_key_info() -> Dict[str, Any]:
    """What GET /v1/public-key returns."""
    raw = public_key_bytes()
    _, ephemeral = _load_key()
    return {
        "algorithm": ALGORITHM,
        "publicKey": _b64(raw),
        "keyId": key_id(raw),
        # Surfaced deliberately. A verifier that does not know the key is
        # throwaway would read a valid signature as more durable than it is.
        "ephemeral": ephemeral,
        "receiptVersion": RECEIPT_VERSION,
    }


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------
def sha256_file_hex(path) -> str:
    """SHA-256 of a file, read in chunks so a 200 MB upload is not held in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decimal(value: Optional[float], places: int = 4) -> Optional[str]:
    """Format a number as a fixed-point decimal string, or None.

    See CANONICAL FORM above: floats are not put in the payload as JSON numbers
    because Python and JavaScript do not always spell them the same way.
    """
    if value is None:
        return None
    return "%.*f" % (places, float(value))


def build(*, analysis_id: str, file_sha256: str, file_name: str,
          file_size: int, verdict: str, confidence: Optional[float],
          model_version: str, analysed_at: str,
          decision_threshold: Optional[float]) -> Dict[str, Any]:
    """Assemble and sign one receipt.

    Every field is copied from the analysis that just ran. Nothing here is
    recomputed or rounded differently from what the report shows, so a receipt
    and the report it came from can never disagree.
    """
    key, ephemeral = _load_key()

    payload: Dict[str, Any] = {
        "receiptVersion": RECEIPT_VERSION,
        "analysisId": analysis_id,
        "fileSha256": file_sha256,
        "fileName": file_name,
        "fileSizeBytes": file_size,
        "verdict": verdict,
        # Decimal strings, not floats - see _decimal and CANONICAL FORM.
        "confidence": _decimal(confidence),
        "decisionThreshold": _decimal(decision_threshold),
        "modelVersion": model_version,
        "analysedAt": analysed_at,
        # Stated inside the signed payload rather than alongside it, so it
        # cannot be stripped off in transit to make a receipt look stronger.
        "attests": ("This detector produced this verdict for the file with this "
                    "SHA-256. It is not a statement that the video is genuine, "
                    "nor that the verdict is correct."),
    }

    signature = key.sign(canonical_bytes(payload))
    return {
        "payload": payload,
        "algorithm": ALGORITHM,
        "keyId": key_id(),
        "ephemeralKey": ephemeral,
        "signature": _b64(signature),
    }


# ---------------------------------------------------------------------------
# Verifying
# ---------------------------------------------------------------------------
def verify(receipt: Dict[str, Any],
           public_key_b64: Optional[str] = None) -> Dict[str, Any]:
    """Check a receipt's signature. Returns a result dict, never raises.

    public_key_b64 defaults to this server's own key, which is the common case
    (verifying a receipt this deployment issued). Passing one explicitly lets
    the same code check a receipt from a different deployment.
    """
    if not isinstance(receipt, dict):
        return {"valid": False, "reason": "Receipt is not a JSON object."}

    payload = receipt.get("payload")
    signature_b64 = receipt.get("signature")
    if not isinstance(payload, dict) or not isinstance(signature_b64, str):
        return {"valid": False,
                "reason": "Receipt must have a 'payload' object and a 'signature' string."}

    if receipt.get("algorithm", ALGORITHM) != ALGORITHM:
        return {"valid": False,
                "reason": "Unsupported algorithm %r; this verifier only does %s."
                          % (receipt.get("algorithm"), ALGORITHM)}

    try:
        raw_public = _unb64(public_key_b64) if public_key_b64 else public_key_bytes()
        public = Ed25519PublicKey.from_public_bytes(raw_public)
    except Exception as exc:                          # noqa: BLE001
        return {"valid": False, "reason": "Unusable public key: %s" % exc}

    try:
        signature = _unb64(signature_b64)
    except Exception as exc:                          # noqa: BLE001
        return {"valid": False, "reason": "Signature is not valid base64: %s" % exc}

    expected_key = key_id(raw_public)
    try:
        public.verify(signature, canonical_bytes(payload))
    except InvalidSignature:
        # The most likely cause by far is that the payload was edited after
        # signing, which is precisely what the receipt exists to detect.
        return {
            "valid": False,
            "reason": ("Signature does not match the payload. Either the "
                       "receipt was altered after signing, or it was signed by "
                       "a different key than the one it was checked against."),
            "keyId": expected_key,
        }
    except Exception as exc:                          # noqa: BLE001
        return {"valid": False, "reason": "Verification failed: %s" % exc}

    result = {
        "valid": True,
        "keyId": expected_key,
        "payload": payload,
    }
    # A receipt naming a different key that still verifies means the caller
    # supplied a key that happens to work; say so rather than hiding it.
    if receipt.get("keyId") and receipt["keyId"] != expected_key:
        result["note"] = ("Receipt names key %s but was verified against %s."
                          % (receipt["keyId"], expected_key))
    if receipt.get("ephemeralKey"):
        result["note"] = ("Signed with a throwaway development key. The "
                          "signature is real, but the key is not a durable "
                          "identity and disappears when the server restarts.")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main(argv) -> int:
    if len(argv) >= 1 and argv[0] == "keygen":
        seed = Ed25519PrivateKey.generate().private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        print("REELREAL_SIGNING_KEY=%s" % _b64(seed))
        print()
        print("Set that in the server's environment. Keep it out of git:")
        print("  Windows : set REELREAL_SIGNING_KEY=...")
        print("  bash    : export REELREAL_SIGNING_KEY=...")
        return 0

    if len(argv) >= 2 and argv[0] == "verify":
        with open(argv[1], "r", encoding="utf-8") as handle:
            receipt = json.load(handle)
        outcome = verify(receipt, argv[2] if len(argv) > 2 else None)
        print(json.dumps(outcome, indent=2))
        return 0 if outcome.get("valid") else 1

    print(__doc__)
    print("Usage:")
    print("  python server/receipt.py keygen")
    print("  python server/receipt.py verify receipt.json [public-key-base64]")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
