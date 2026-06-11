"""Content entropy filters for the evolution/mutation pipeline.

Blocks look-alike code bloat and parameter recycling by measuring
structural diversity in recipe text before accepting mutations.
"""

import hashlib
import math
import re
from collections import Counter


# ---------------------------------------------------------------------------
# Entropy metrics
# ---------------------------------------------------------------------------

def token_entropy(text: str) -> float:
    """Shannon entropy of whitespace-split tokens in *text*.
    Higher values indicate more diverse content."""
    tokens = text.split()
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
    )


def line_entropy(text: str) -> float:
    """Shannon entropy of non-empty lines. Detects line-level duplication."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    total = len(lines)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
    )


def param_fingerprint(text: str) -> str:
    """Deterministic fingerprint of all numeric/string parameters in the text.
    Identical fingerprints across mutations signal parameter recycling."""
    params = re.findall(r'"[^"]*"|\b\d+\b', text)
    blob = "|".join(sorted(params))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def structural_diversity(text: str) -> float:
    """Ratio of unique task bodies to total tasks. 1.0 = fully unique."""
    blocks = re.split(r"(?=\[task:)", text)
    bodies = []
    for b in blocks:
        lines = [l.strip() for l in b.split("\n") if l.strip() and not l.strip().startswith("[task:") and not l.strip().startswith("needs =")]
        if lines:
            bodies.append("\n".join(lines))
    if not bodies:
        return 1.0
    unique = len(set(bodies))
    return unique / len(bodies)


# ---------------------------------------------------------------------------
# Filter decisions
# ---------------------------------------------------------------------------

_MIN_TOKEN_ENTROPY = 3.0
_MIN_LINE_ENTROPY = 2.5
_MIN_STRUCTURAL_DIVERSITY = 0.4


def check_entropy(recipe_text: str, *,
                  min_token_entropy: float = _MIN_TOKEN_ENTROPY,
                  min_line_entropy: float = _MIN_LINE_ENTROPY,
                  min_diversity: float = _MIN_STRUCTURAL_DIVERSITY) -> dict:
    """Evaluate a recipe and return pass/fail with metrics.

    Returns a dict with keys:
        passed: bool
        token_entropy: float
        line_entropy: float
        structural_diversity: float
        param_fingerprint: str
        reasons: list[str]  (non-empty if failed)
    """
    te = token_entropy(recipe_text)
    le = line_entropy(recipe_text)
    sd = structural_diversity(recipe_text)
    pf = param_fingerprint(recipe_text)
    reasons = []

    if te < min_token_entropy:
        reasons.append(f"Token entropy {te:.2f} below threshold {min_token_entropy}")
    if le < min_line_entropy:
        reasons.append(f"Line entropy {le:.2f} below threshold {min_line_entropy}")
    if sd < min_diversity:
        reasons.append(f"Structural diversity {sd:.2f} below threshold {min_diversity}")

    return {
        "passed": len(reasons) == 0,
        "token_entropy": round(te, 3),
        "line_entropy": round(le, 3),
        "structural_diversity": round(sd, 3),
        "param_fingerprint": pf,
        "reasons": reasons,
    }


def detect_param_recycling(old_text: str, new_text: str) -> bool:
    """Return True if the mutation recycled the same parameter set."""
    return param_fingerprint(old_text) == param_fingerprint(new_text)
