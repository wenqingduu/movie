"""Deterministic prompt gates for local identity injection.

The 3D identity reference is reconstructed from a neutral, open-eye portrait.
When a shot explicitly requests closed eyes, injecting that reference into the
eye region creates a direct expression conflict.  For now the conservative
policy is to reuse Control for the whole shot.
"""

from __future__ import annotations

import re


PROMPT_EYE_CLOSURE_SKIP_REASON = "prompt_eye_closure_conflict"


_CHINESE_NEGATED_EYE_CLOSURE = re.compile(
    r"(?:不|别|勿|未|没有|无需|不要|不能|不可|避免)\s*"
    r"(?:慢慢|缓缓|轻轻|紧紧)?\s*"
    r"(?:闭(?:上|紧)?(?:双)?眼(?:睛)?|合上(?:双)?眼(?:睛)?|闭目|阖眼)"
)
_CHINESE_EYE_CLOSURE = re.compile(
    r"(?:"
    r"(?:慢慢|缓缓|轻轻|紧紧)?\s*闭(?:上|紧)?(?:双)?眼(?:睛)?"
    r"|(?:慢慢|缓缓|轻轻|紧紧)?\s*合上(?:双)?眼(?:睛)?"
    r"|(?:双)?眼(?:睛)?\s*(?:缓缓|慢慢|轻轻|紧紧)?\s*(?:闭上|紧闭|闭合)"
    r"|闭目|阖眼"
    r")"
)

_ENGLISH_NEGATED_EYE_CLOSURE = re.compile(
    r"\b(?:do(?:es)?\s+not|did\s+not|must\s+not|should\s+not|never|without)\s+"
    r"(?:slowly\s+|gently\s+|tightly\s+)?"
    r"(?:close|closes|closed|closing|shut|shuts|shutting)"
    r"(?:\s+\w+){0,3}\s+eyes?\b"
    r"|\beyes?\s+(?:are\s+|is\s+|remain\s+|stays?\s+)?not\s+(?:closed|shut)\b",
    re.IGNORECASE,
)
_ENGLISH_EYE_CLOSURE = re.compile(
    r"\b(?:close|closes|closed|closing|shut|shuts|shutting)"
    r"(?:\s+\w+){0,3}\s+eyes?\b"
    r"|\beyes?\s+(?:are\s+|is\s+|remain\s+|stays?\s+)?(?:closed|shut)\b",
    re.IGNORECASE,
)


def prompt_requests_closed_eyes(prompt: str | None) -> bool:
    """Return True only for an explicit, non-negated closed-eye instruction."""

    if not prompt:
        return False
    sanitized = _CHINESE_NEGATED_EYE_CLOSURE.sub(" ", prompt)
    sanitized = _ENGLISH_NEGATED_EYE_CLOSURE.sub(" ", sanitized)
    return bool(
        _CHINESE_EYE_CLOSURE.search(sanitized)
        or _ENGLISH_EYE_CLOSURE.search(sanitized)
    )
