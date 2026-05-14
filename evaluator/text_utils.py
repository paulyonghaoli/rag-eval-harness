from __future__ import annotations

import re
from typing import List


def split_into_claims(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Split on ", and " only when followed by a subject pronoun — that pattern
    # marks independent-clause conjunctions ("…glucose, and it occurs in Y") but
    # not Oxford-comma enumerations ("refracted, dispersed, and reflected").
    fragments = re.split(
        r"[.?!;]\s*"
        r"|,\s+and\s+(?=it\b|they\b|this\b|these\b|its\b|their\b|he\b|she\b|we\b|you\b)",
        text,
        flags=re.IGNORECASE,
    )
    claims = []
    for fragment in fragments:
        fragment = fragment.strip().rstrip(".")
        if len(fragment.split()) < 2:
            continue
        claims.append(fragment)
    return claims or [text]
