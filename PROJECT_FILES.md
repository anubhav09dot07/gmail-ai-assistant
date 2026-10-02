# Project Summary and File Guide

## What this project does

AI Gmail Knowledge Assistant is a Gmail-grounded question-answering application.
It authenticates with Google OAuth, reads the user's Gmail messages, extracts and
normalizes message content, chunks and embeds the content, stores vectors in
Qdrant, and answers Gmail-specific questions with evidence-backed Groq/LangChain
generation. The React frontend provides sign-in, Gmail previews, chat, loading
states, errors, and source links back to Gmail.

The central safety rule is: no relevant Gmail evidence means no generated answer.
Queries are scoped to the authenticated user's data, and retrieved email content
is treated as untrusted evidence rather than instructions.

## Runtime flow

1. The React client checks authentication status and starts Google sign-in.
2. FastAPI handles OAuth, stores the local token, and creates Gmail API clients.
3. Gmail messages are fetched, MIME-decoded, cleaned, normalized, and chunked.
4. Local embeddings are stored in user-filtered Qdrant payloads.
5. A chat query is planned and normalized, then searched using Gmail operators,
   lexical matching, and vector retrieval as appropriate.
6. Candidates are filtered, fused, reranked, deduplicated, and checked against
   the evidence threshold.
7. LangChain/Groq receives bounded Gmail evidence and returns an answer with
   metadata-only citations.

## File guide

### Documentation and project configuration

- `README.md`: setup, local commands, configuration, endpoints, ingestion,
  retrieval, sync, and RAG notes.
- `PROJECT_OVERVIEW.md`: product definition, architecture goals, security rules,
  and future roadmap.
- `PROJECT_FILES.md`: this implementation-focused file map.
- `flow.odt`: editable project flow/design document.
- `.gitignore`: excludes credentials, tokens, environments, caches, generated
  output, and local data.
- `backend/.env.example`: safe configuration template; copy it to `backend/.env`
  for local use.
- `backend/requirements.txt`: backend Python dependencies.
- `backend/requirements-cpu.txt`: CPU-oriented installation requirements.
- `frontend/package.json`: frontend scripts and dependencies.
- `frontend/package-lock.json`: locked frontend dependency graph.

### Backend application

- `backend/app/main.py`: FastAPI app, middleware, auth, email, sync, chat, and
  error-shaped API endpoints.
- `backend/app/config.py`: environment loading and application defaults.
- `backend/app/oauth.py`: Google OAuth flow, token loading, persistence, and
  logout cleanup.
- `backend/app/gmail.py`: Gmail API clients, pagination, message retrieval,
  headers, dates, labels, and normalized message conversion.
- `backend/app/gmail_sync_service.py`: capped mailbox sync, fingerprints,
  idempotent updates, changed-message replacement, and safe reconciliation.
- `backend/app/mime.py`: recursive MIME parsing, base64 decoding, and HTML/plain
  text extraction.
- `backend/app/preprocessing.py`: whitespace cleanup and display-only link
  normalization.
- `backend/app/models.py`: normalized Gmail message and API data models.
- `backend/app/chunking.py`: paragraph/sentence-aware chunking with overlap,
  metadata retention, and deterministic IDs.
- `backend/app/embeddings.py`: lazy Sentence Transformers embedding generation.
- `backend/app/vector_store.py`: Qdrant collection management, filtering,
  upserts, search, scrolling, and user-scoped deletion.
- `backend/app/query_planner.py`: Gmail relevance gate and structured query
  constraint extraction.
- `backend/app/query_normalizer.py`: aliases, abbreviations, and corpus-aware
  typo normalization.
- `backend/app/query_service.py`: retrieval orchestration, filters, scoring,
  evidence decisions, grouping, and safe failure behavior.
- `backend/app/response.py`: normalized API response formatting.
- `backend/app/groq_service.py`: direct Groq SDK boundary and error handling.
- `backend/app/rag_service.py`: grounded answer orchestration and citations.

### Query, RAG, retrieval, and LLM packages

- `backend/app/query/schemas.py`: typed Gmail query and retrieval-plan models.
- `backend/app/query/router.py`: maps query intent to retrieval modes.
- `backend/app/query/understanding.py`: deterministic understanding with optional
  structured-output LLM fallback.
- `backend/app/rag/prompts.py`: evidence serialization and grounding prompts.
- `backend/app/rag/chains.py`: LangChain generation and structured-search chains.
- `backend/app/rag/citations.py`: public source records and Gmail URLs.
- `backend/app/rag/evidence.py`: evidence validation helpers.
- `backend/app/retrieval/fusion.py`: reciprocal-rank fusion across retrieval
  channels.
- `backend/app/retrieval/reranker.py`: sender, entity, subject, and recency
  reranking.
- `backend/app/retrieval/qdrant_retriever.py`: LangChain retriever adapter for
  user-scoped Qdrant search.
- `backend/app/llm/factory.py`: configurable LangChain chat-model factory.

### Frontend

- `frontend/index.html`: Vite HTML entry point.
- `frontend/src/main.jsx`: React app, auth status, chat submission, Gmail
  previews, source links, logout, loading, and error states.
- `frontend/src/styles.css`: responsive visual layout and component styling.

### Tests

The tests under `backend/tests/` cover the chat API, chunking, display cleaning,
embeddings, ingestion, sync, hybrid and structured retrieval, LangChain RAG,
OAuth file permissions, Qdrant behavior, query normalization, query service,
grounding, regressions, and tenant isolation.

## Files intentionally excluded from Git

Never publish `backend/.env`, the root `.env`, `backend/token.json`, the Google
`client_secret*.json` file, sync metadata, virtual environments, `node_modules`,
frontend build output, caches, logs, or local vector/database artifacts. These
may contain API keys, OAuth client secrets, Gmail refresh tokens, private email
metadata, or generated machine-specific data.

The credentials currently present in the local workspace should be revoked or
rotated before using this project outside the local machine.