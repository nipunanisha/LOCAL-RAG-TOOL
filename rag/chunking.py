"""
Semantic chunking — section-aware, sentence-aware, token-budgeted.

A "section" is a heading-bounded region. Within a section we pack sentences
until the token budget is hit, then start a new chunk with N sentences of
overlap. Sections are never crossed by a single chunk.

Each input is a list of "blocks": dicts with at least {"text": str, "meta": dict}.
Headings/page-breaks/slide-breaks become natural block boundaries with
meta hints like {"section": "Q1 results", "page": 3, "slide": 2, "sheet": "X"}.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable

# tiktoken is optional — char-approx fallback works fine
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def _tokens(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))
except Exception:  # pragma: no cover
    def _tokens(text: str) -> int:
        # rough: 1 token ≈ 4 chars for english
        return max(1, len(text) // 4)


# Sentence splitter — keeps abbreviations like "U.S." and "Mr." intact
_ABBREVS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "inc", "ltd", "co",
    "u.s", "u.k", "e.g", "i.e", "etc", "vs", "no", "fig", "eq",
}
_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+"           # after sentence-ending punct + whitespace
    r"(?=[A-Z\"'(\[])"         # before capital / opening quote
)


def _is_abbrev_end(text: str) -> bool:
    """Return True if `text` ends with a known abbreviation (so the period
    that follows is NOT a sentence boundary). Avoids variable-width lookbehind
    which Python's re module doesn't support."""
    m = re.search(r"([A-Za-z.]+)[.!?]?$", text)
    if not m:
        return False
    tok = m.group(1).lower().rstrip(".")
    return tok in _ABBREVS


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # also break on double-newlines (paragraph breaks)
    parts: list[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        raw = _SENT_BOUNDARY.split(para)
        merged: list[str] = []
        for s in raw:
            s = s.strip()
            if not s:
                continue
            if merged and _is_abbrev_end(merged[-1]):
                merged[-1] = merged[-1] + " " + s
            else:
                merged.append(s)
        parts.extend(merged)
    return parts


@dataclass
class Block:
    text: str
    meta: dict   # may carry: section, page, slide, sheet


@dataclass
class Chunk:
    text: str
    meta: dict   # carries section/page/slide/sheet from the block(s) it came from
    token_count: int


def chunk_blocks(
    blocks: Iterable[Block],
    target_tokens: int = 350,
    min_tokens: int = 50,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Pack sentences from blocks into chunks. Never cross block boundaries."""
    out: list[Chunk] = []

    for block in blocks:
        sentences = split_sentences(block.text)
        if not sentences:
            continue

        buf: list[str] = []
        buf_tokens = 0

        def flush(carry: list[str] | None = None):
            nonlocal buf, buf_tokens
            if not buf:
                return
            text = " ".join(buf).strip()
            if text:
                out.append(Chunk(text=text, meta=dict(block.meta), token_count=buf_tokens))
            buf = list(carry) if carry else []
            buf_tokens = sum(_tokens(s) for s in buf)

        for sent in sentences:
            t = _tokens(sent)
            # sentence alone exceeds target — emit it as its own chunk
            if t > target_tokens:
                flush()
                out.append(Chunk(text=sent, meta=dict(block.meta), token_count=t))
                continue

            if buf_tokens + t > target_tokens and buf:
                # flush, then carry last N sentences as overlap
                carry = buf[-overlap_sentences:] if overlap_sentences else []
                flush(carry=carry)

            buf.append(sent)
            buf_tokens += t

        flush()

    # merge tiny tail chunks into their predecessor (within same section)
    merged: list[Chunk] = []
    for c in out:
        if merged and c.token_count < min_tokens and merged[-1].meta.get("section") == c.meta.get("section"):
            prev = merged[-1]
            prev.text = prev.text + " " + c.text
            prev.token_count += c.token_count
        else:
            merged.append(c)

    return merged
