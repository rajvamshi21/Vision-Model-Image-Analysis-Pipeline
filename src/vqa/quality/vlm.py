"""Optional multimodal critique layer.

The heuristics say *that* an image is soft; a VLM can say *why* it will not
convert ("the label is illegible and the reflection hides the cap"). Providers
are pluggable and failures degrade to ``VLMAssessment(error=...)`` so a flaky
API never takes down a batch.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from abc import ABC, abstractmethod

from PIL import Image

from vqa.config import Settings, get_settings
from vqa.types import VLMAssessment

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a strict e-commerce catalogue reviewer. Judge whether a product photo "
    "is ready to publish on a marketplace listing."
)

USER_PROMPT = """Assess this product photo and reply with JSON only:

{
  "score": <0-100 overall publish-readiness>,
  "caption": "<one sentence describing the product>",
  "tags": ["<3-6 short attribute tags>"],
  "issues": ["<specific, actionable problems; empty list if none>"]
}

Judge focus, lighting, background cleanliness, framing, colour accuracy and
anything that would confuse or mislead a buyer."""


def encode_jpeg(image: Image.Image, max_side: int = 768, quality: int = 85) -> str:
    """Downscale + base64-encode: VLM cost scales with pixels, not with detail."""
    image = image.convert("RGB")
    if max(image.size) > max_side:
        ratio = max_side / float(max(image.size))
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_response(text: str, provider: str, model: str) -> VLMAssessment:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return VLMAssessment(provider=provider, model=model, raw=text,
                             error="no JSON object in response")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return VLMAssessment(provider=provider, model=model, raw=text, error=str(exc))

    score = payload.get("score")
    return VLMAssessment(
        provider=provider,
        model=model,
        score=float(score) if isinstance(score, (int, float)) else None,
        caption=payload.get("caption"),
        tags=[str(t) for t in payload.get("tags", [])][:8],
        issues=[str(i) for i in payload.get("issues", [])][:8],
        raw=text,
    )


class VLMCritic(ABC):
    provider = "none"

    @abstractmethod
    def assess(self, image: Image.Image) -> VLMAssessment | None: ...


class NullCritic(VLMCritic):
    """Default: no network calls, no keys, no cost."""

    def assess(self, image: Image.Image) -> VLMAssessment | None:
        return None


class AnthropicCritic(VLMCritic):
    provider = "anthropic"

    def __init__(self, model: str, max_tokens: int = 700) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def assess(self, image: Image.Image) -> VLMAssessment:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64",
                                                     "media_type": "image/jpeg",
                                                     "data": encode_jpeg(image)}},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                }],
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            return parse_response(text, self.provider, self.model)
        except Exception as exc:
            logger.warning("Anthropic critique failed: %s", exc)
            return VLMAssessment(provider=self.provider, model=self.model, error=str(exc))


class OpenAICritic(VLMCritic):
    provider = "openai"

    def __init__(self, model: str, max_tokens: int = 700) -> None:
        import openai

        self.client = openai.OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def assess(self, image: Image.Image) -> VLMAssessment:
        try:
            data_url = f"data:image/jpeg;base64,{encode_jpeg(image)}"
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ],
            )
            return parse_response(response.choices[0].message.content or "",
                                  self.provider, self.model)
        except Exception as exc:
            logger.warning("OpenAI critique failed: %s", exc)
            return VLMAssessment(provider=self.provider, model=self.model, error=str(exc))


def get_critic(settings: Settings | None = None) -> VLMCritic:
    settings = settings or get_settings()
    provider = settings.vlm_provider
    try:
        if provider == "anthropic":
            return AnthropicCritic(settings.vlm_model, settings.vlm_max_tokens)
        if provider == "openai":
            return OpenAICritic(settings.vlm_model, settings.vlm_max_tokens)
    except Exception as exc:  # pragma: no cover - depends on optional deps/keys
        logger.warning("VLM provider %r unavailable (%s); continuing without it", provider, exc)
        return NullCritic()
    return NullCritic()
