from __future__ import annotations

import re
from typing import List


def split_into_claims(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    fragments = re.split(r"[.?!;]\s*", text)
    claims = []
    for fragment in fragments:
        fragment = fragment.strip().rstrip(".")
        if len(fragment.split()) < 2:
            continue
        claims.append(fragment)
    return claims or [text]
