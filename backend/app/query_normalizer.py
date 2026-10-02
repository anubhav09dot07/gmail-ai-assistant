from __future__ import annotations

import difflib
import re
from collections.abc import Iterable


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
_ALIAS_REPLACEMENTS = {
    "mail": "email",
    "mails": "emails",
    "msg": "message",
    "msgs": "messages",
    "signin": "sign-in",
    "acct": "account",
    "acc": "account",
    "pwd": "password",
    "sub": "subscription",
}


def normalize_query(query: str) -> str:
    """Normalize Gmail vocabulary without changing the user's original query."""
    normalized = query.lower().replace("\u2010", "-").replace("\u2011", "-")
    normalized = re.sub(r"\b(can not|can't)\b", "cannot", normalized)
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = [_ALIAS_REPLACEMENTS.get(token, token) for token in _TOKEN_PATTERN.findall(normalized)]
    return " ".join(tokens)


def _corpus_terms(corpus: Iterable[str]) -> set[str]:
    terms: set[str] = set()
    for value in corpus:
        terms.update(
            token
            for token in _TOKEN_PATTERN.findall(value.lower())
            if token.isalpha()
        )
    return terms


def _replace_supported_aliases(normalized: str, vocabulary: set[str]) -> str:
    aliases = {"ig": "instagram", "insta": "instagram"}
    return " ".join(
        aliases.get(token, token) if aliases.get(token, token) in vocabulary else token
        for token in normalized.split()
    )


def _correct_tokens(normalized: str, corpus: Iterable[str]) -> str:
    vocabulary = _corpus_terms(corpus)
    if not vocabulary:
        return normalized
    corrected: list[str] = []
    for token in normalized.split():
        if token in vocabulary or len(token) < 5:
            corrected.append(token)
            continue
        match = difflib.get_close_matches(token, vocabulary, n=1, cutoff=0.88)
        corrected.append(match[0] if match else token)
    return " ".join(corrected)


def expand_query(query: str, corpus: Iterable[str] = ()) -> tuple[str, ...]:
    """Return a small deterministic set of normalized and corpus-corrected variants."""
    normalized = normalize_query(query)
    corpus_values = tuple(corpus)
    vocabulary = _corpus_terms(corpus_values)
    alias_expanded = _replace_supported_aliases(normalized, vocabulary)
    corrected = _correct_tokens(alias_expanded, corpus_values)
    variants: list[str] = []
    for variant in (query.strip(), normalized, alias_expanded, corrected):
        if variant and variant not in variants:
            variants.append(variant)
    return tuple(variants[:5])
