"""
Dev tool: caption a dataset directory with Claude Sonnet (ADR-0004).

Walks a local directory of images, sends each to Claude Sonnet with the
describe_identity prompt (tools/prompts/describe_identity.md), and emits:

  - captions.json   (filename -> caption; consumed by the handler's `captions` field)
  - (optional) <image_stem>.txt alongside each image, for manual train.py runs

Prompt caching: the system prompt is sent as a cached block so the first
image eats the cache cost once, subsequent images pay read-only rates.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m tools.caption_dataset \
        --images ./my_dataset/ \
        --trigger TOK \
        [--out ./my_dataset/captions.json] \
        [--write-txt] [--dry-run] [--model claude-sonnet-4-5]

Phase 1: run manually before calling the RunPod handler.
Phase 2: avatar-backend imports this module as a library.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("caption")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Resolve the prompt relative to this file so the tool works from any CWD.
_PROMPT_PATH = Path(__file__).parent / "prompts" / "describe_identity.md"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _load_system_prompt() -> str:
    """
    Return the system-prompt body — everything under the `## System prompt`
    heading, up to the next `---` separator in describe_identity.md.
    """
    text = _PROMPT_PATH.read_text()
    marker = "## System prompt"
    idx = text.find(marker)
    if idx < 0:
        raise RuntimeError(f"missing '{marker}' in {_PROMPT_PATH}")
    tail = text[idx + len(marker):]
    end = tail.find("\n---")
    return tail[:end].strip() if end >= 0 else tail.strip()


def _b64_image(path: Path) -> tuple[str, str]:
    """Return (media_type, base64_data). Raises on unsupported extension."""
    mime, _ = mimetypes.guess_type(path.name)
    if mime not in ("image/jpeg", "image/png", "image/webp"):
        raise ValueError(f"unsupported image type for {path}: {mime}")
    return mime, base64.standard_b64encode(path.read_bytes()).decode("ascii")


def caption_image(
    client: Any, *,
    image_path: Path, trigger: str, model: str, system_prompt: str,
) -> str:
    """
    One-shot caption call. System prompt is cached (`cache_control` block).
    The image + user text is the uncached "message" portion.
    """
    media_type, data = _b64_image(image_path)
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type, "data": data,
                }},
                {"type": "text",
                 "text": f"Trigger token: {trigger}\nCaption this photo."},
            ],
        }],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    caption = "".join(parts).strip()
    if not caption:
        raise RuntimeError(f"empty caption for {image_path.name}")
    return caption


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True,
                    help="Directory of training images")
    ap.add_argument("--trigger", required=True, help="e.g. 'TOK'")
    ap.add_argument("--out", type=Path, default=None,
                    help="Path for captions.json (default: <images>/captions.json)")
    ap.add_argument("--model", default="claude-sonnet-4-5",
                    help="Anthropic model id")
    ap.add_argument("--write-txt", action="store_true",
                    help="Also write <stem>.txt next to each image "
                         "(for manual train.py runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List images but do not call the API")
    args = ap.parse_args()

    if not args.images.is_dir():
        log.error("images dir not found: %s", args.images)
        return 2

    images = sorted([
        p for p in args.images.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and not p.name.startswith(".")
    ])
    if not images:
        log.error("no images in %s", args.images)
        return 2
    log.info("found %d images", len(images))

    out_path = args.out or (args.images / "captions.json")
    existing: dict[str, str] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            log.info("resuming from %s (%d captions)", out_path, len(existing))
        except json.JSONDecodeError:
            log.warning("existing %s malformed — ignoring", out_path)

    if args.dry_run:
        for p in images:
            log.info("would caption: %s", p.name)
        return 0

    # Import only when actually calling the API — keeps `--dry-run` dep-free
    try:
        import anthropic  # type: ignore
    except ImportError:
        log.error("pip install anthropic  # required unless --dry-run")
        return 2

    system_prompt = _load_system_prompt()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY env var

    captions = dict(existing)
    for i, img in enumerate(images, 1):
        if img.name in captions:
            log.info("[%d/%d] skip (cached): %s", i, len(images), img.name)
            continue
        try:
            caption = caption_image(
                client, image_path=img, trigger=args.trigger,
                model=args.model, system_prompt=system_prompt,
            )
        except Exception as e:
            log.error("[%d/%d] %s failed: %s", i, len(images), img.name, e)
            # Persist what we have so a re-run resumes
            out_path.write_text(json.dumps(captions, indent=2, sort_keys=True))
            return 3
        captions[img.name] = caption
        log.info("[%d/%d] %s -> %s", i, len(images), img.name, caption[:80])
        if args.write_txt:
            (img.with_suffix(".txt")).write_text(caption)
        # Persist incrementally so a crash never loses work
        out_path.write_text(json.dumps(captions, indent=2, sort_keys=True))

    log.info("wrote %s (%d captions)", out_path, len(captions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
