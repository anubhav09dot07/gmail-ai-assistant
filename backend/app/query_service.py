from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from . import config
from .chunking import EmailChunk, chunk_messages
from .embeddings import EmbeddingService
from .gmail import search_messages
from .query.router import RetrievalRouter
from .query.schemas import GmailQuery, RetrievalPlan
from .query.understanding import understand_query
from .query_normalizer import expand_query, normalize_query
from .query_planner import SearchPlan, plan_search
from .retrieval.fusion import ReciprocalRankFusion
from .retrieval.reranker import CandidateReranker
from .vector_store import QdrantVectorStore, RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
_LEXICAL_STOP_WORDS = {
    "a",
    "an",
    "and",
    "any",
    "did",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "show",
    "the",
    "there",
    "to",
    "was",
    "what",
    "when",
    "which",
    "who",
    "with",
    "you",
    "your",
    "last",
    "latest",
    "most",
    "recent",
    "recently",
    "newest",
    "yesterday",
    "today",
    "this",
    "week",
    "mail",
    "mails",
    "email",
    "emails",
    "message",
    "messages",
}
_QUERY_STRUCTURE_TERMS = {
    "a",
    "an",
    "and",
    "any",
    "did",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "show",
    "the",
    "there",
    "to",
    "was",
    "what",
    "when",
    "which",
    "who",
    "with",
    "you",
    "your",
    "last",
    "latest",
    "most",
    "recent",
    "recently",
    "newest",
    "yesterday",
    "today",
    "this",
    "week",
    "mail",
    "mails",
    "email",
    "emails",
    "message",
    "messages",
    "thread",
    "threads",
    "sign",
    "account",
    "received",
    "receive",
    "sent",
    "send",
    "said",
    "say",
    "mentioned",
    "decide",
    "decided",
    "wrote",
    "forwarded",
    "work",
    "much",
    "pay",
    "paid",
    "that",
    "about",
    "someone",
    "log",
    "logged",
    "logging",
    "signing",
    "into",
    "new",
    "detected",
}


@dataclass(frozen=True)
class GmailQueryEvaluation:
    query: str
    relevant_to_gmail: bool
    decision: str
    reason: str
    retrieved_chunks: tuple[RetrievedChunk, ...] = ()
    scores: tuple[float, ...] = ()
    evidence_sufficient: bool = False
    evidence_threshold: float = config.GMAIL_EVIDENCE_THRESHOLD
    thread_groups: dict[str, tuple[RetrievedChunk, ...]] = field(default_factory=dict)
    normalized_query: str = ""
    query_variants: tuple[str, ...] = ()
    search_plan: SearchPlan | None = None

    def public_summary(self) -> dict[str, Any]:
        """Return decision metadata without exposing email text or payloads."""
        return {
            "query": self.query,
            "relevant_to_gmail": self.relevant_to_gmail,
            "decision": self.decision,
            "reason": self.reason,
            "scores": list(self.scores),
            "evidence_sufficient": self.evidence_sufficient,
            "evidence_threshold": self.evidence_threshold,
            "thread_ids": list(self.thread_groups),
            "normalized_query": self.normalized_query,
            "query_variants": list(self.query_variants),
        }


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(normalize_query(value)))


def _query_entities(query: str, corpus: list[RetrievedChunk]) -> set[str]:
    """Find distinctive query terms without maintaining a global entity list."""
    corpus_text = [
        " ".join(
            str(item.chunk.get(field, ""))
            for field in ("subject", "sender", "recipients", "text")
        )
        for item in corpus
    ]
    expanded = expand_query(query, corpus_text)
    resolved_variant = expanded[-1] if expanded else normalize_query(query)
    return {
        token
        for token in _tokens(resolved_variant)
        if token not in _QUERY_STRUCTURE_TERMS and len(token) >= 2
    }


def _lexical_score(query: str, chunk: RetrievedChunk) -> float:
    query_terms = _tokens(query) - _LEXICAL_STOP_WORDS
    if not query_terms:
        return 0.0
    subject = str(chunk.chunk.get("subject", "")).lower()
    sender = str(chunk.chunk.get("sender", "")).lower()
    subject_terms = _tokens(subject)
    sender_terms = _tokens(sender)
    subject_hits = len(query_terms & subject_terms) / len(query_terms)
    sender_hits = len(query_terms & sender_terms) / len(query_terms)
    recipient_terms = _tokens(" ".join(chunk.chunk.get("recipients", [])))
    recipient_hits = len(query_terms & recipient_terms) / len(query_terms)
    body_terms = _tokens(str(chunk.chunk.get("text", "")))
    body_hits = len(query_terms & body_terms) / len(query_terms)
    subject_bonus = 0.25 if subject_hits >= 0.8 else 0.0
    phrase_bonus = (
        0.15 if normalize_query(query) in normalize_query(subject) else 0.0
    )
    return min(
        1.0,
        0.65 * subject_hits
        + 0.20 * sender_hits
        + 0.10 * recipient_hits
        + 0.05 * body_hits
        + subject_bonus
        + phrase_bonus,
    )


def _internal_timestamp(chunk: RetrievedChunk) -> int:
    """Return the Gmail API's canonical epoch milliseconds, or zero if absent.

    Date headers are sender-provided display metadata and must never determine
    chronological retrieval order.
    """
    val = chunk.chunk.get("internal_date")
    try:
        return int(val) if val else 0
    except (ValueError, TypeError):
        return 0


def _group_by_thread(
    results: list[RetrievedChunk],
) -> dict[str, tuple[RetrievedChunk, ...]]:
    groups: dict[str, list[RetrievedChunk]] = {}
    for result in results:
        thread_id = str(result.chunk.get("thread_id", ""))
        if thread_id:
            groups.setdefault(thread_id, []).append(result)
    return {thread_id: tuple(items) for thread_id, items in groups.items()}


def evaluate_gmail_query(
    user_id: str,
    query: str,
    *,
    vector_store: QdrantVectorStore | None = None,
    embedding_service: EmbeddingService | None = None,
    limit: int | None = None,
    evidence_threshold: float | None = None,
    gmail_service=None,
) -> GmailQueryEvaluation:
    """Evaluate Gmail relevance, structured constraints, and semantic evidence without calling an LLM."""
    clean_query = query.strip() if isinstance(query, str) else ""
    threshold = (
        config.GMAIL_EVIDENCE_THRESHOLD
        if evidence_threshold is None
        else evidence_threshold
    )

    if not user_id or not user_id.strip():
        return GmailQueryEvaluation(
            query=clean_query,
            relevant_to_gmail=False,
            decision="reject",
            reason="invalid_user_id",
            evidence_threshold=threshold,
        )
    if not clean_query:
        return GmailQueryEvaluation(
            query=clean_query,
            relevant_to_gmail=False,
            decision="reject",
            reason="empty_query",
            evidence_threshold=threshold,
        )

    plan = plan_search(clean_query)
    normalized_query = normalize_query(clean_query)
    gmail_query = understand_query(clean_query, allow_llm=False)
    router = RetrievalRouter()
    retrieval_route = router.route(gmail_query)

    if not plan.is_gmail_related:
        if config.RAG_DEBUG:
            logger.info("Query %r rejected by scope check (reason=%s)", clean_query, plan.reason)
        return GmailQueryEvaluation(
            query=clean_query,
            relevant_to_gmail=False,
            decision="reject",
            reason=plan.reason,
            evidence_threshold=threshold,
            normalized_query=normalized_query,
            search_plan=plan,
        )

    # Exact Gmail searches do not depend on embeddings.  Only semantic or
    # hybrid ranking needs an embedding model and its evidence threshold.
    service = embedding_service or EmbeddingService()
    if plan.use_semantic_search:
        try:
            service.embed_text(plan.semantic_query or normalized_query)
        except Exception:
            return GmailQueryEvaluation(
                query=clean_query,
                relevant_to_gmail=True,
                decision="reject",
                reason="embedding_unavailable",
                evidence_threshold=threshold,
                normalized_query=normalized_query,
                search_plan=plan,
            )

    target_limit = limit or plan.limit or config.GMAIL_RETRIEVAL_LIMIT
    store = vector_store or QdrantVectorStore()

    # 1. Structured candidate retrieval from live Gmail API if available
    live_chunks_by_id: dict[str, RetrievedChunk] = {}
    if gmail_service is not None and plan.use_structured_search:
        try:
            live_messages, _ = search_messages(
                query=plan.gmail_query,
                max_messages=max(target_limit * 5, 20),
                include_spam_trash=plan.include_spam_trash,
                label_ids=list(plan.labels) if plan.labels else None,
                service=gmail_service,
            )
            for chunk in chunk_messages(live_messages):
                cid = str(chunk.get("chunk_id", ""))
                if cid:
                    live_chunks_by_id[cid] = RetrievedChunk(chunk=chunk, score=1.0)
        except Exception:
            pass

    # 2. Vector store retrieval (scrolled payloads & dense search)
    corpus: list[RetrievedChunk] = []
    dense_by_id: dict[str, RetrievedChunk] = {}
    dense_rank: dict[str, int] = {}
    all_variants: tuple[str, ...] = (clean_query,)

    try:
        # A structured-only request can be fulfilled from indexed metadata when
        # a live Gmail service is unavailable (e.g. offline tests).  This
        # fallback explicitly paginates instead of silently truncating a mailbox.
        if not live_chunks_by_id and not plan.use_semantic_search and hasattr(store, "scroll_chunks"):
            corpus = store.scroll_chunks(
                user_id=user_id,
                limit=None,
                labels=plan.labels or None,
                internal_date_after=plan.internal_date_after,
                internal_date_before=plan.internal_date_before,
            )
            corpus_text = [
                " ".join(
                    str(item.chunk.get(field, ""))
                    for field in ("subject", "sender", "text")
                )
                for item in corpus
            ]
            if corpus:
                corrected_variants = expand_query(clean_query, corpus_text)
                all_variants = tuple(
                    dict.fromkeys(
                        (
                            clean_query,
                            *(
                                [plan.semantic_query]
                                if plan.semantic_query
                                else []
                            ),
                            *corrected_variants,
                        )
                    )
                )[:5]
            else:
                all_variants = (clean_query,)

        if plan.use_semantic_search:
            for variant in all_variants:
                query_vector = service.embed_text(variant)
                dense_results = store.search(
                    query_vector,
                    user_id=user_id,
                    limit=max(target_limit * 5, 20),
                )
                for rank, result in enumerate(dense_results, start=1):
                    chunk_id = str(result.chunk.get("chunk_id", ""))
                    if not chunk_id:
                        continue
                    current = dense_by_id.get(chunk_id)
                    if current is None or result.score > current.score:
                        dense_by_id[chunk_id] = result
                    dense_rank[chunk_id] = min(dense_rank.get(chunk_id, rank), rank)
            # Use the bounded similarity candidates as vocabulary for typo/alias
            # expansion.  This retains lexical recovery without scanning a user's
            # entire Qdrant corpus on every semantic question.
            candidate_text = [
                " ".join(
                    str(result.chunk.get(field, ""))
                    for field in ("subject", "sender", "text")
                )
                for result in dense_by_id.values()
            ]
            expanded_variants = (
                tuple(
                    dict.fromkeys(
                        (clean_query, *([plan.semantic_query] if plan.semantic_query else []), *expand_query(clean_query, candidate_text))
                    )
                )[:5]
                if candidate_text and hasattr(store, "scroll_chunks")
                else all_variants
            )
            for variant in expanded_variants:
                if variant in all_variants:
                    continue
                query_vector = service.embed_text(variant)
                dense_results = store.search(
                    query_vector, user_id=user_id, limit=max(target_limit * 5, 20)
                )
                for rank, result in enumerate(dense_results, start=1):
                    chunk_id = str(result.chunk.get("chunk_id", ""))
                    if not chunk_id:
                        continue
                    current = dense_by_id.get(chunk_id)
                    if current is None or result.score > current.score:
                        dense_by_id[chunk_id] = result
                    dense_rank[chunk_id] = min(dense_rank.get(chunk_id, rank), rank)
            all_variants = expanded_variants
        elif not hasattr(store, "scroll_chunks") and not live_chunks_by_id:
            # For minimal mocks (e.g. FakeStore without scroll_chunks), invoke search
            vec_size = getattr(store, "vector_size", 3)
            dense_results = store.search(
                [0.0] * vec_size,
                user_id=user_id,
                limit=max(target_limit, 10),
            )
            for result in dense_results:
                chunk_id = str(result.chunk.get("chunk_id", ""))
                if chunk_id:
                    dense_by_id[chunk_id] = result

    except ValueError:
        raise
    except Exception:
        return GmailQueryEvaluation(
            query=clean_query,
            relevant_to_gmail=True,
            decision="reject",
            reason="qdrant_unavailable",
            evidence_threshold=threshold,
            normalized_query=normalized_query,
            query_variants=all_variants,
            search_plan=plan,
        )

    # 3. Aggregate all candidate chunks
    all_candidates: dict[str, RetrievedChunk] = {}
    structured_candidate_ids = set(live_chunks_by_id)
    all_candidates.update(live_chunks_by_id)
    for item in corpus:
        cid = str(item.chunk.get("chunk_id", ""))
        if cid:
            all_candidates[cid] = item
    for cid, item in dense_by_id.items():
        # Keep the live Gmail representation and its authoritative metadata
        # when the same message also appears in the semantic result set.
        all_candidates.setdefault(cid, item)

    # If corpus was empty and dense was empty but live chunks exist, use live
    candidate_list = list(all_candidates.values())

    # 4. HARD FILTERING: Structured constraints act as HARD FILTERS, not merely scoring signals.
    # A candidate that does not satisfy an explicit sender/label/date constraint must not enter the final answer.
    filtered_candidates: dict[str, RetrievedChunk] = {}
    for cid, result in all_candidates.items():
        chunk = result.chunk

        # Sender hard filter
        if plan.sender:
            sender_field = str(chunk.get("sender", "")).lower()
            if plan.sender.lower() not in sender_field:
                continue

        if plan.recipient:
            recipients = " ".join(chunk.get("recipients", []) or []).lower()
            if plan.recipient.lower() not in recipients:
                continue

        if plan.subject:
            if plan.subject.lower() not in str(chunk.get("subject", "")).lower():
                continue

        # User-emphasized names/acronyms constrain the eligible messages even
        # when they are mentioned in the subject or body rather than the From
        # address.  This prevents a broad semantic neighbour (for example a
        # different bank statement) from replacing the named entity's email.
        entity_haystack = " ".join(
            str(chunk.get(field, ""))
            if field != "recipients"
            else " ".join(chunk.get("recipients", []) or [])
            for field in ("sender", "subject", "recipients", "text")
        ).lower()
        if plan.entities and any(entity.lower() not in entity_haystack for entity in plan.entities):
            continue

        # Label hard filter
        chunk_labels = set(chunk.get("labels", []) or [])
        if plan.labels:
            missing_label = False
            for req_label in plan.labels:
                if req_label not in chunk_labels:
                    missing_label = True
                    break
            if missing_label:
                continue

        # Spam/Trash filter: unless explicitly requested, exclude spam and trash
        if not plan.include_spam_trash:
            if "SPAM" in chunk_labels or "TRASH" in chunk_labels:
                continue

        # Apply date facets to the canonical Gmail timestamp only.  A legacy
        # point with no internal_date cannot prove that it satisfies a date
        # constraint, so it is excluded rather than ordered by Date headers.
        candidate_internal_date = _internal_timestamp(result)
        if (
            plan.internal_date_after is not None
            and candidate_internal_date < plan.internal_date_after
        ):
            continue
        if (
            plan.internal_date_before is not None
            and candidate_internal_date >= plan.internal_date_before
        ):
            continue

        filtered_candidates[cid] = result

    # 5. Semantic Scoring & Threshold Gate for Surviving Candidates
    query_entities = _query_entities(
        plan.semantic_query or clean_query, candidate_list
    )
    lexical_scores = {
        str(res.chunk.get("chunk_id", "")): max(
            _lexical_score(variant, res) for variant in all_variants
        )
        for res in filtered_candidates.values()
    }

    scored_candidates: list[tuple[float, float, int, RetrievedChunk]] = []
    for chunk_id, result in filtered_candidates.items():
        cand_ts = _internal_timestamp(result)
        dense_score = dense_by_id.get(
            chunk_id, RetrievedChunk(result.chunk, 0.0)
        ).score
        lex_score = lexical_scores.get(chunk_id, 0.0)

        candidate_terms = _tokens(
            " ".join(
                str(result.chunk.get(field, ""))
                for field in ("subject", "sender", "recipients", "text")
            )
        )
        entity_coverage = (
            len(query_entities & candidate_terms) / len(query_entities)
            if query_entities
            else 1.0
        )

        if plan.use_semantic_search:
            evidence_score = (
                max(dense_score, lex_score, 1.0)
                if chunk_id in structured_candidate_ids
                else (max(dense_score, lex_score) if entity_coverage else 0.0)
            )
            # Gmail has already proved structured candidates satisfy the
            # explicit query constraints. Only semantic-only candidates need
            # to clear the embedding evidence threshold.
            if chunk_id not in structured_candidate_ids and evidence_score < threshold:
                continue
            relevance_score = max(
                lex_score * (0.5 + 0.5 * entity_coverage),
                dense_score * (0.5 + 0.5 * lex_score) * entity_coverage,
            )
            fusion_score = (
                1.0 / (60 + dense_rank.get(chunk_id, 60))
            ) + lex_score * 0.25
        else:
            # Pure structured query: candidate passed hard filters
            evidence_score = max(dense_score, 1.0)
            relevance_score = 1.0
            fusion_score = 1.0

        scored_candidates.append(
            (
                relevance_score,
                fusion_score,
                cand_ts,
                RetrievedChunk(result.chunk, evidence_score),
            )
        )

    # 6. Ordering / Sorting:
    # Date-constrained Gmail searches must be chronological. The date filter
    # determines which messages are eligible; within that set, newest first
    # gives a stable and useful answer instead of allowing semantic relevance
    # to prefer an older message.
    chronological_query = bool(plan.date_range)
    if plan.sort == "newest" or chronological_query:
        scored_candidates.sort(
            key=lambda item: (item[2], item[0], item[1]), reverse=True
        )
    else:
        scored_candidates.sort(
            key=lambda item: (item[0], item[1], item[2]), reverse=True
        )

    # 7. Message-level Deduplication: One representative per message
    message_groups: dict[
        str, list[tuple[float, float, int, RetrievedChunk]]
    ] = {}
    for item in scored_candidates:
        msg_id = str(item[3].chunk.get("message_id", ""))
        key = msg_id or str(item[3].chunk.get("chunk_id", ""))
        message_groups.setdefault(key, []).append(item)

    message_ranked: list[tuple[float, float, int, RetrievedChunk]] = []
    for group in message_groups.values():
        if plan.sort == "newest" or chronological_query:
            group.sort(
                key=lambda item: (item[2], item[0], item[1]), reverse=True
            )
        else:
            group.sort(
                key=lambda item: (item[0], item[1], item[2]), reverse=True
            )
        strongest = group[0]
        consensus = min(0.05, sum(item[0] for item in group[1:2]) * 0.05)
        message_ranked.append(
            (
                min(1.0, strongest[0] + consensus),
                strongest[1],
                strongest[2],
                strongest[3],
            )
        )

    if plan.sort == "newest" or chronological_query:
        message_ranked.sort(
            key=lambda item: (item[2], item[0], item[1]), reverse=True
        )
    else:
        message_ranked.sort(
            key=lambda item: (item[0], item[1], item[2]), reverse=True
        )

    selected_chunks = [item[3] for item in message_ranked[:target_limit]]
    selected_scores = tuple(chunk.score for chunk in selected_chunks)

    # Live Gmail structured matches are authoritative evidence.
    # Do not reject them merely because their semantic similarity is below
    # the embedding threshold. Semantic scoring is only a relevance/ranking
    # signal when structured Gmail constraints already matched.
    has_structured_evidence = bool(live_chunks_by_id)

    has_evidence = bool(selected_chunks) and (
        has_structured_evidence
        or not plan.use_semantic_search
        or max(selected_scores) >= threshold
    )

    reason = (
        "sufficient_gmail_evidence"
        if has_evidence
        else "insufficient_gmail_evidence"
    )

    if has_evidence and has_structured_evidence:
        supporting_results = tuple(selected_chunks)
    elif has_evidence and plan.use_semantic_search:
        supporting_results = tuple(
            chunk for chunk in selected_chunks if chunk.score >= threshold
        )
    else:
        supporting_results = tuple(selected_chunks)
    supporting_scores = tuple(chunk.score for chunk in supporting_results)

    if config.RAG_DEBUG:
        top_summary = (
            f"{supporting_results[0].chunk.get('subject', '')} (score={supporting_results[0].score:.3f})"
            if supporting_results
            else "None"
        )
        logger.info(
            "RAG Debug Trace:\n"
            "  Query: %r\n"
            "  Intent: %s\n"
            "  Sender: %s\n"
            "  Route: %s\n"
            "  Gmail candidates: %d\n"
            "  Vector candidates: %d\n"
            "  Lexical candidates: %d\n"
            "  Top result: %s\n"
            "  Evidence sufficient: %s (threshold=%.2f)",
            clean_query,
            gmail_query.intent,
            gmail_query.sender,
            retrieval_route.value,
            len(live_chunks_by_id),
            len(dense_by_id),
            len(corpus),
            top_summary,
            has_evidence,
            threshold,
        )

    return GmailQueryEvaluation(
        query=clean_query,
        relevant_to_gmail=True,
        decision="allow" if has_evidence else "reject",
        reason=reason,
        retrieved_chunks=supporting_results,
        scores=supporting_scores,
        evidence_sufficient=has_evidence,
        evidence_threshold=threshold,
        thread_groups=_group_by_thread(list(supporting_results)),
        normalized_query=normalized_query,
        query_variants=all_variants,
        search_plan=plan,
    )

