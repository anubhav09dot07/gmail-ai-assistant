# AI Gmail Knowledge Assistant

Phase 1 establishes the Google OAuth -> Gmail API connection. Phase 2 adds Gmail message ingestion, MIME-aware body extraction, preprocessing, and normalized email metadata. Phase 3 adds metadata-preserving chunking and local embeddings.

## Prerequisites

- Python 3.11 or newer
- Node.js 20 or newer
- A Google Cloud project with the Gmail API enabled
- An OAuth 2.0 Web application client configured with:
  - Authorized JavaScript origin: `http://localhost:5173`
  - Authorized redirect URI: `http://localhost:8000/api/auth/callback`

## Configure OAuth

Copy `backend/.env.example` to `backend/.env` and set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and a long random `SESSION_SECRET`. Keep `backend/.env` private. The backend stores the OAuth token in `backend/token.json`, which is also ignored by Git.

## Run the backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
cd ..
uvicorn backend.app.main:app --reload --port 8000
```

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, select **Connect Gmail**, approve the read-only Gmail permission, and return to the app. The page then calls `GET /api/emails` and displays up to 10 normalized messages with metadata and a cleaned body preview.

## Phase 1 API

- `GET /api/health` - backend health check
- `GET /api/auth/login` - begins Google OAuth
- `GET /api/auth/callback` - completes OAuth and redirects to the frontend
- `GET /api/auth/status` - reports whether a local Gmail token is available
- `POST /api/auth/logout` - removes the local token
- `GET /api/emails?limit=10` - fetches and normalizes a small Gmail message batch

## Phase 2 Ingestion

The ingestion layer separates Gmail API fetching, MIME/body extraction, text preprocessing, and response formatting. It supports nested multipart messages, prefers `text/plain`, converts HTML-only messages to readable text while retaining useful URLs, and preserves message/thread IDs, recipients, timestamps, snippets, and labels.

Focused tests are in `backend/tests/test_ingestion.py` and cover plain text, nested multipart, HTML-only, and empty-body messages.

## Phase 3 Chunking and Embeddings

Email bodies are split into paragraph and sentence-aware chunks, with a hard character split only when one unit is larger than the configured budget. Each chunk includes a compact subject/sender/recipient context plus the original message and thread metadata. Empty email bodies produce no chunks.

Defaults are `CHUNK_SIZE=1200` characters and `CHUNK_OVERLAP=200` characters. This keeps typical email evidence below a manageable context size while carrying enough neighboring text across boundaries for names, dates, amounts, and URLs. Both values are configurable in `backend/.env`.

Embeddings are generated locally through Sentence Transformers using `BAAI/bge-small-en-v1.5`, whose output dimension is 384. The `EmbeddingService` loads the model lazily, batches chunk text in one `encode` call, normalizes vectors, and validates finite numeric output. No email content or vectors are sent to an external generative API.

The backend environment uses CPU-only PyTorch (`torch==2.14.0+cpu`) from the official PyTorch CPU index. This avoids pulling CUDA packages for local development. The model weights are downloaded from Hugging Face on first real embedding use and cached by the Hugging Face client outside the repository.

## Phase 4 Qdrant Vector Storage

Phase 4 stores the Phase 3 chunks and local embeddings in the existing persistent Qdrant service at `QDRANT_URL` (default `http://localhost:6333`). The adapter uses the `gmail_knowledge` collection, 384-dimensional normalized vectors, cosine distance, batched upserts, deterministic UUID point IDs, and payload metadata for user, message, thread, sender, recipients, subject, date, labels, chunk ID, and cleaned text.

Use `QdrantVectorStore.index_messages(messages, embedding_service, user_id=...)` to chunk, embed, and index a normalized Gmail batch. Use `search(query_vector, user_id=..., sender=..., subject=..., label=..., date_from=..., date_to=..., thread_id=...)` for similarity retrieval. Every retrieval and deletion is user-scoped at the Qdrant filter layer.

## Phase 5 Gmail Relevance and Evidence Gates

`evaluate_gmail_query(user_id, query)` is a retrieval-only service. A deterministic, explainable local classifier first rejects clearly unrelated questions before embedding or Qdrant access. It also accepts contextual questions such as “What did they finally decide?” so the evidence gate, rather than a keyword-only rule, decides whether Gmail supports the request.

Gmail-related queries are embedded locally and searched with the mandatory user-scoped Qdrant filter. Results are allowed only when the best cosine score meets `GMAIL_EVIDENCE_THRESHOLD`, which defaults to `0.62`; otherwise the service rejects with `insufficient_gmail_evidence`. Empty queries, invalid users, embedding failures, Qdrant failures, and no results fail closed. Results retain scores and can be grouped by `thread_id` for later bounded thread expansion.

The initial threshold is based on the Phase 4 calibration observations: a clearly unrelated query scored about `0.51`, while Gmail-oriented queries scored about `0.66` to `0.71`. This is an initial operating point, not a universal relevance guarantee; it must be recalibrated as the mailbox, embedding model, and query distribution grow. **No Gmail evidence = no answer.**

## Phase 6 Grounded RAG and Groq

`generate_gmail_answer(user_id, query)` calls Phase 5 first. Rejected or weakly supported queries return a structured rejection and never call Groq. Approved chunks are placed into a compact, bounded context with explicit evidence delimiters and source metadata. Email content is untrusted data; instructions inside an email cannot override the system grounding instructions.

Groq is isolated in `GroqService`, uses the official `groq` SDK, and reads `GROQ_API_KEY`, `GROQ_MODEL`, and `GROQ_TIMEOUT_SECONDS` from `backend/.env`. The default model is `openai/gpt-oss-120b`, but it remains configurable. Responses are rejected when the API key is missing, the request fails, or the model returns empty content. The service returns the grounded answer with subject, sender, date, message ID, thread ID, and similarity score sources. It does not expose email bodies, API keys, or prompt contents in public result metadata.

Groq is not a general-purpose chatbot: it receives only approved Gmail evidence and must not use outside knowledge. Internal evidence delimiters are removed from the final user-facing answer, while email chunks remain available internally for grounding. Public source attribution is deduplicated by `message_id` and keeps the strongest chunk score; distinct messages in the same thread remain distinct. **No Gmail evidence = no answer.** The frontend chat UI and later application wiring remain outside Phase 6.

The current 56-point validation mailbox contains payment-related bank and Google Play messages, but no university, semester, tuition, MCA, college, institute, or academic-fee evidence. The semester-fee query therefore remains correctly fail-closed; retrieval improvements must be calibrated only after the relevant Gmail message is indexed.

## Phase 8 Robust Query Retrieval

Query retrieval keeps a small deterministic Gmail safety gate, then normalizes email terms such as `mail`, `msg`, and `signin`. Abbreviations such as `IG` and close typos such as `instagarm`, `securty`, and `statment` are expanded only when the authenticated user's indexed payload vocabulary supports the target term; the original query is always retained.

The existing dense Qdrant vectors remain available without reindexing. Retrieval also scans the authenticated user's existing payloads for bounded lexical matches across subject, sender, and chunk text. Dense and lexical candidates are fused locally, with temporal wording such as `latest`, `last`, and `recent` preferring newer messages only after relevance has been considered. The evidence decision uses a separate evidence score and the existing `GMAIL_EVIDENCE_THRESHOLD`; reciprocal-rank-style fusion is never compared directly with that threshold.

This improves discovery of exact senders, subjects, identifiers, abbreviations, and common typos without making Groq responsible for relevance. Groq still receives only approved Gmail evidence, and rejected or insufficient-evidence results expose no source candidates. No evidence still means no answer, and unrelated questions still stop before retrieval and generation.

## Phase 8.2 Gmail Sync

`POST /api/sync` runs an authenticated, user-scoped mailbox sync. Gmail message listing uses page tokens with `GMAIL_SYNC_PAGE_SIZE` (default `100`) and stops at `GMAIL_SYNC_MAX_MESSAGES` (default `1000`). The existing `GET /api/emails` endpoint remains a small display fetch and is not the production indexing path.

Sync metadata is stored atomically in `backend/sync_metadata.json` by default. Each authenticated user has message-level fingerprints keyed by stable Gmail `message_id`; unchanged messages are skipped, changed messages have their old chunks deleted before replacement, and deterministic chunk IDs make repeated upserts idempotent. A complete listing reconciles missing message IDs and deletes only that user's chunks. A capped or otherwise partial listing never treats unseen messages as deleted. No Qdrant collection recreation or bulk reindex is used.

## Phase 9 Hybrid LangChain RAG Architecture

The RAG pipeline has been upgraded with a hybrid architecture combining LangChain abstractions with Gmail-specific retrieval, strict gating, and multi-tenant isolation.

### Architecture Diagram

```text
                         USER QUERY
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Query Understanding  │
                  │   Fast-path / LLM    │
                  └──────────┬───────────┘
                             │
                  Structured GmailQuery
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Retrieval Router   │
                  └──────────┬───────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
     Gmail Search         Qdrant           Metadata
     / Gmail API        Vector Search       Search
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                    Candidate Documents
                             │
                             ▼
                  Reciprocal Rank Fusion
                             │
                             ▼
                     Candidate Reranker
                             │
                             ▼
                    Evidence Validation
                             │
                ┌────────────┴────────────┐
                │                         │
          insufficient                sufficient
                │                         │
                ▼                         ▼
        Safe Gmail refusal        LangChain Grounded
                                      Generation
                                           │
                                           ▼
                                 Answer + Gmail Citations
```

### Key Architectural Tenets

1. **Why LangChain is Used**:
   - Standardizes ChatModel abstractions (`ChatGroq` / configurable providers via `backend/app/llm/factory.py`).
   - Clean prompt templating via `ChatPromptTemplate` with anti-prompt-injection grounding system instructions.
   - Structured query understanding (`with_structured_output`) generating typed `GmailQuery` objects.
   - Standard `BaseRetriever` interface (`GmailQdrantRetriever`) wrapping Qdrant with guaranteed user isolation.
   - Output parsing and runnable orchestration (`prompt | chat_model | StrOutputParser`).

2. **Why Direct Gmail Search is Retained**:
   - Vector search alone cannot reliably answer queries like *"When was the last email from Instagram?"* or *"Show unread receipts from this week"*.
   - Direct Gmail search via `search_messages` queries authoritative headers, labels, and boundaries (`from:`, `after:`, `before:`, `label:`) directly from Gmail's index.

3. **Why Qdrant is Retained**:
   - Provides local, low-latency dense vector search using normalized `BAAI/bge-small-en-v1.5` embeddings (384 dimensions) with payload filtering.
   - Fully isolated per-user (`user_id` payload filter) with zero external vector hosting costs.

4. **Why Vector Search Alone is Insufficient**:
   - Semantic embeddings prioritize topical similarity rather than chronological recency or exact sender matching. A semantically similar message from 2 years ago must not beat an exact sender message from yesterday when the user asks for the *"latest"* email.

5. **How Temporal Queries Work**:
   - Temporal words (`latest`, `last`, `recent`, `yesterday`, `last week`) are extracted into `sort_order = "newest"` and explicit `time_start`/`time_end` epoch milliseconds.
   - Temporal rank weighting in `ReciprocalRankFusion` and `CandidateReranker` boosts messages based on canonical `internal_date`.

6. **How Hallucinations & Injections are Prevented**:
   - **Gate 1 (Scope)**: Non-Gmail questions (trivia, coding, weather) are immediately rejected before vector search or LLM generation.
   - **Gate 2 (Evidence)**: If retrieved chunks do not clear `GMAIL_EVIDENCE_THRESHOLD` (default `0.62`), the system fails closed: *"I couldn't find enough information in your Gmail to answer that."*
   - **Untrusted Data Rule**: Email content is enclosed inside explicit `[BEGIN GMAIL EVIDENCE]` delimiters and labeled as untrusted data that cannot override system instructions.

7. **Multi-Tenant User Isolation**:
   - `GmailQdrantRetriever` strictly requires a non-empty `user_id`. Every Qdrant similarity search and scroll query applies a hard boolean filter matching `user_id`, preventing any cross-user data leakage.