"""Zero-shot quality attributes from a CLIP-style joint embedding space.

Each attribute is a contrastive prompt pair. We score an image by softmaxing
its cosine similarity against the two prompts, which turns "does this look like
a professional studio shot?" into a calibrated probability without any labelled
training data. Prompt embeddings are computed once and cached.
"""

from __future__ import annotations

import numpy as np

from vqa.embedding.base import Encoder
from vqa.types import SemanticAttributes

# attribute -> (positive prompt, negative prompt)
PROMPT_PAIRS: dict[str, tuple[str, str]] = {
    "in_focus": (
        "a sharp, perfectly in-focus product photograph",
        "a blurry, out-of-focus, smeared photograph",
    ),
    "professional_lighting": (
        "a product photo with even professional studio lighting",
        "a photo with harsh shadows, dim or uneven lighting",
    ),
    "clean_background": (
        "a product isolated on a clean plain white background",
        "a product photographed in a messy cluttered room",
    ),
    "well_framed": (
        "a centred, well-framed product filling the frame",
        "a badly cropped photo with the product tiny or cut off",
    ),
    "studio_quality": (
        "a high quality professional e-commerce catalogue photo",
        "a low quality amateur snapshot taken on an old phone",
    ),
    "has_watermark": (
        "a photograph with a visible watermark or logo overlay",
        "a clean photograph with no text or watermark",
    ),
    "lifestyle_shot": (
        "a lifestyle photo of a product being used in a real scene",
        "a plain packshot of a product on a seamless background",
    ),
}

LOGIT_SCALE = 100.0  # CLIP's learned temperature; standard for zero-shot use


class ZeroShotAttributeScorer:
    def __init__(self, encoder: Encoder, prompt_pairs: dict[str, tuple[str, str]] | None = None):
        if not encoder.supports_text:
            raise ValueError(
                f"{encoder.name} has no text tower; zero-shot attributes require CLIP/SigLIP"
            )
        self.encoder = encoder
        self.prompt_pairs = prompt_pairs or PROMPT_PAIRS
        self.keys = list(self.prompt_pairs)
        flat = [p for key in self.keys for p in self.prompt_pairs[key]]
        self._text = encoder.encode_texts(flat)          # (2 * n_attributes, dim)

    def score(self, embeddings: np.ndarray) -> list[SemanticAttributes]:
        """``embeddings``: (n, dim) L2-normalised image vectors."""
        embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        logits = LOGIT_SCALE * embeddings @ self._text.T        # (n, 2 * n_attributes)
        pairs = logits.reshape(embeddings.shape[0], len(self.keys), 2)
        pairs = pairs - pairs.max(axis=-1, keepdims=True)       # numerically safe softmax
        exp = np.exp(pairs)
        probabilities = exp[..., 0] / exp.sum(axis=-1)
        return [
            SemanticAttributes(
                model=self.encoder.name,
                scores={key: round(float(row[i]), 4) for i, key in enumerate(self.keys)},
            )
            for row in probabilities
        ]


def maybe_build_scorer(encoder: Encoder) -> ZeroShotAttributeScorer | None:
    """Return a scorer, or ``None`` when the encoder has no text tower."""
    if not encoder.supports_text:
        return None
    return ZeroShotAttributeScorer(encoder)
