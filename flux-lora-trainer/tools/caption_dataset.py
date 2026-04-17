"""
Dev tool: caption a dataset directory with a VLM (ADR-0004).

Walks a local directory of images, sends each to a VLM with the
describe_identity prompt (tools/prompts/describe_identity.md), and emits:

  - captions.json   (filename -> caption; consumed by the handler's `captions` field)
  - (optional) <image_stem>.txt alongside each image, for manual train.py runs

**Provider fallback chain.** Per image, providers are tried in order until one
returns a valid caption. Default order: ``claude,openai,gemini``. Override
with ``--providers``. Fallback is per-image: if Anthropic rate-limits on
image 3 of 6, that one image can be captioned by GPT-5 while images 1,2,4,5,6
still come from Claude. The provider is stored alongside the caption in an
audit log so a quality regression can be attributed to a specific provider.

Prompt caching: the system prompt is sent as a cached block to Claude
(``cache_control: ephemeral``); OpenAI and Gemini don't expose an equivalent
single-turn cache control so each call pays the full system-prompt tokens.
That's fine — the chain only falls through on Claude failure anyway.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...        # optional (for fallback)
    export GEMINI_API_KEY=...           # optional (for fallback)

    python -m tools.caption_dataset \\
        --images ./my_dataset/ \\
        --trigger TOK \\
        [--providers claude,openai,gemini] \\
        [--out ./my_dataset/captions.json] \\
        [--write-txt] [--dry-run]

Phase 1: run manually before calling the RunPod handler.
Phase 2: avatar-backend imports this module as a library.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("caption")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Resolve the prompt relative to this file so the tool works from any CWD.
_PROMPT_PATH = Path(__file__).parent / "prompts" / "describe_identity.md"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Default model per provider. Bump these when users confirm newer SKUs are
# available on their account. claude-sonnet-4-5 = current Anthropic flagship.
DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-5",
    "openai": "gpt-5.3",
    "gemini": "gemini-2.5-pro",
}


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


# ─────────────────────────────────────────────────────────────────────────────
# Provider adapters. Each returns the caption string or raises.
# ─────────────────────────────────────────────────────────────────────────────
def _caption_claude(
    image_path: Path, *, trigger: str, model: str, system_prompt: str,
) -> str:
    import anthropic  # type: ignore
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic()
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
        raise RuntimeError("empty caption")
    return caption


def _caption_openai(
    image_path: Path, *, trigger: str, model: str, system_prompt: str,
) -> str:
    from openai import OpenAI  # type: ignore
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI()
    media_type, data = _b64_image(image_path)
    # Responses API pattern — same chat-style shape works for vision.
    resp = client.chat.completions.create(
        model=model,
        max_tokens=200,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:{media_type};base64,{data}",
                }},
                {"type": "text",
                 "text": f"Trigger token: {trigger}\nCaption this photo."},
            ]},
        ],
    )
    caption = (resp.choices[0].message.content or "").strip()
    if not caption:
        raise RuntimeError("empty caption")
    return caption


def _caption_gemini(
    image_path: Path, *, trigger: str, model: str, system_prompt: str,
) -> str:
    # google-genai is the newer SDK (2025+); fall back to google-generativeai
    # if the user still has the older one pinned.
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
                    or os.environ["GOOGLE_API_KEY"],
        )
        media_type, _ = _b64_image(image_path)
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(
                    data=image_path.read_bytes(), mime_type=media_type,
                ),
                f"Trigger token: {trigger}\nCaption this photo.",
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=200,
            ),
        )
        caption = (resp.text or "").strip()
    except ImportError:
        import google.generativeai as genai  # type: ignore
        genai.configure(
            api_key=os.environ.get("GEMINI_API_KEY")
                    or os.environ["GOOGLE_API_KEY"],
        )
        media_type, data = _b64_image(image_path)
        m = genai.GenerativeModel(
            model_name=model, system_instruction=system_prompt,
        )
        resp = m.generate_content([
            {"mime_type": media_type, "data": base64.b64decode(data)},
            f"Trigger token: {trigger}\nCaption this photo.",
        ])
        caption = (resp.text or "").strip()
    if not caption:
        raise RuntimeError("empty caption")
    return caption


PROVIDERS: dict[str, Callable[..., str]] = {
    "claude": _caption_claude,
    "openai": _caption_openai,
    "gemini": _caption_gemini,
}


def caption_with_fallback(
    image_path: Path, *,
    trigger: str, system_prompt: str,
    providers: list[str], models: dict[str, str],
) -> tuple[str, str]:
    """
    Try providers in order; return (caption, provider_name_that_succeeded).
    Raises the last error only if *all* providers fail.
    """
    last_err: Exception | None = None
    for name in providers:
        fn = PROVIDERS[name]
        try:
            caption = fn(
                image_path,
                trigger=trigger,
                model=models[name],
                system_prompt=system_prompt,
            )
            return caption, name
        except Exception as e:  # noqa: BLE001 — fallback is the whole point
            last_err = e
            log.warning("  provider=%s failed: %s", name, e)
    raise RuntimeError(f"all providers failed for {image_path.name}: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True,
                    help="Directory of training images")
    ap.add_argument("--trigger", required=True, help="e.g. 'TOK'")
    ap.add_argument("--out", type=Path, default=None,
                    help="Path for captions.json (default: <images>/captions.json)")
    ap.add_argument("--providers", default="claude,openai,gemini",
                    help="Comma-separated fallback chain. Default: claude,openai,gemini")
    ap.add_argument("--claude-model", default=DEFAULT_MODELS["claude"])
    ap.add_argument("--openai-model", default=DEFAULT_MODELS["openai"])
    ap.add_argument("--gemini-model", default=DEFAULT_MODELS["gemini"])
    ap.add_argument("--write-txt", action="store_true",
                    help="Also write <stem>.txt next to each image "
                         "(for manual train.py runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List images but do not call any API")
    args = ap.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    unknown = [p for p in providers if p not in PROVIDERS]
    if unknown:
        log.error("unknown provider(s): %s (known: %s)",
                  unknown, list(PROVIDERS))
        return 2
    models = {
        "claude": args.claude_model,
        "openai": args.openai_model,
        "gemini": args.gemini_model,
    }

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
    log.info("found %d images; provider chain: %s",
             len(images), "->".join(providers))

    out_path = args.out or (args.images / "captions.json")
    audit_path = out_path.with_name(out_path.stem + ".audit.json")
    existing: dict[str, str] = {}
    audit: dict[str, str] = {}  # filename -> provider
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            log.info("resuming from %s (%d captions)", out_path, len(existing))
        except json.JSONDecodeError:
            log.warning("existing %s malformed — ignoring", out_path)
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text())
        except json.JSONDecodeError:
            pass

    if args.dry_run:
        for p in images:
            log.info("would caption: %s", p.name)
        return 0

    system_prompt = _load_system_prompt()

    captions = dict(existing)
    for i, img in enumerate(images, 1):
        if img.name in captions:
            log.info("[%d/%d] skip (cached): %s", i, len(images), img.name)
            continue
        try:
            caption, provider = caption_with_fallback(
                img, trigger=args.trigger, system_prompt=system_prompt,
                providers=providers, models=models,
            )
        except Exception as e:
            log.error("[%d/%d] %s failed: %s", i, len(images), img.name, e)
            # Persist what we have so a re-run resumes
            out_path.write_text(json.dumps(captions, indent=2, sort_keys=True))
            audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
            return 3
        captions[img.name] = caption
        audit[img.name] = provider
        log.info("[%d/%d] %s (via %s) -> %s",
                 i, len(images), img.name, provider, caption[:80])
        if args.write_txt:
            (img.with_suffix(".txt")).write_text(caption)
        # Persist incrementally so a crash never loses work
        out_path.write_text(json.dumps(captions, indent=2, sort_keys=True))
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True))

    log.info("wrote %s (%d captions)", out_path, len(captions))
    log.info("wrote %s (provider audit)", audit_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
