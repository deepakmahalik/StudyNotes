"""
Guardrails for the RAG pipeline.

Checks applied
--------------
1. Input validation   — reject empty or excessively long queries
2. Bias / compliance  — block known harmful keyword patterns
3. Hallucination guard — warn when the answer references things not in context
4. Output sanitisation — strip any leaked system-prompt artefacts
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

MAX_QUERY_LEN = 2000
MIN_QUERY_LEN = 2

# Patterns that trigger an immediate refusal (lowercase match)
_BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive|malware|virus))\b",
        r"\b(child\s+(porn|abuse|sexual))\b",
        r"\b(doxx(ing)?|doxing)\b",
        r"\b(self[\-\s]harm|suicide\s+method)\b",
    ]
]

# Phrases that suggest hallucination (answer references things not in context)
_HALLUCINATION_SIGNALS = [
    "according to my knowledge",
    "as of my training",
    "i know that",
    "generally speaking",
    "in general",
    "typically",
]


@dataclass
class GuardrailResult:
    passed: bool = True
    reason: str = ""
    flags: list[str] = field(default_factory=list)
    answer: str = ""


# ── Public API ────────────────────────────────────────────────────────────────

def validate_query(query: str) -> GuardrailResult:
    """Check the user query before sending it to the pipeline."""
    if not query or len(query.strip()) < MIN_QUERY_LEN:
        return GuardrailResult(passed=False, reason="Query is too short.")

    if len(query) > MAX_QUERY_LEN:
        return GuardrailResult(
            passed=False,
            reason=f"Query exceeds maximum length of {MAX_QUERY_LEN} characters.",
        )

    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(query):
            logger.warning("Blocked query matching pattern: %s", pattern.pattern)
            return GuardrailResult(
                passed=False,
                reason="Your query contains content that cannot be processed.",
            )

    return GuardrailResult(passed=True, answer=query)


def check_output(answer: str, context_chunks: list[dict]) -> GuardrailResult:
    """
    Post-generation check on the LLM answer.

    Flags hallucination signals when the answer seems to go beyond context.
    Strips leaked system-prompt artefacts.
    Does NOT block the answer — only adds flags and optionally appends a disclaimer.
    """
    flags: list[str] = []
    cleaned = answer

    # 1. Strip potential prompt leakage
    cleaned = re.sub(r"(?i)(system[\s\-]*prompt|context provided below).*", "", cleaned).strip()

    # 2. Detect hallucination signals
    lower = cleaned.lower()
    for signal in _HALLUCINATION_SIGNALS:
        if signal in lower:
            flags.append(f"possible_hallucination: '{signal}'")

    # 3. If no context chunks were retrieved but the model gave a detailed answer, flag it
    if not context_chunks and len(cleaned) > 200:
        flags.append("no_context_chunks: answer may be fabricated")

    if flags:
        logger.warning("Output guardrail flags: %s", flags)
        cleaned += (
            "\n\n⚠️ *Note: Some parts of this answer may go beyond the stored study notes. "
            "Please verify with additional sources.*"
        )

    return GuardrailResult(passed=True, flags=flags, answer=cleaned)


def sanitise_sources(sources: list[str]) -> list[str]:
    """Remove empty, duplicate, or obviously invalid source strings."""
    seen: set[str] = set()
    clean: list[str] = []
    for s in sources:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            clean.append(s)
    return clean
