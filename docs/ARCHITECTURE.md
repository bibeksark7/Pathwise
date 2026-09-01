# Pathwise — Architecture & Implementation Plan

## Context

`C:\Users\bsark\Downloads\Projects\Pathwise` is empty. This is a greenfield build of an adaptive learning platform whose differentiator is **an adaptive engine, not an LLM wrapper**: a persistent knowledge graph, a deterministic mastery model, a deterministic next-topic decision engine, and a validated retrieval layer — with the LLM confined to interpretation, generation, and explanation.

The design principle that shapes every decision below (spec §17): **deterministic logic decides, the LLM explains.** Graph traversal, mastery math, candidate ranking, and roadmap mutation are pure Python functions with unit tests. The LLM never invents a resource URL, never computes a mastery score, and never picks the next topic — it renders a machine-produced `DecisionTrace` into prose, and that prose is validated against the trace before it is stored.

**Decisions confirmed with you:**
- **DB:** Docker Desktop + `docker-compose` (Postgres 16 + pgvector, Redis). You install Docker Desktop once; nothing runs until then.
- **LLM:** Anthropic (`claude-opus-5`) as the primary provider. Anthropic has no embeddings API, so embeddings come from **fastembed** (`BAAI/bge-small-en-v1.5`, 384-dim, ONNX — no torch, small image). An OpenAI adapter and a deterministic **fake provider** are written behind the same interface; the fake provider is what CI and all tests use, so the suite runs offline and costs nothing.
- **Resources:** curated seed catalog in versioned YAML, HTTP-validated at ingest. LLM ranks and explains only.

---

## 1. System Architecture

```
React (Vite/TS/Tailwind/React Flow)
        │  REST + SSE
        ▼
FastAPI  ── /api/routes ──► /services (business logic, no HTTP types)
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
  Deterministic core      AI layer              Persistence
  ─────────────────      ─────────             ───────────
  graph traversal        LLMProvider iface     Postgres 16
  mastery math           prompt registry       + pgvector
  decision engine        validators/repair     SQLAlchemy 2 async
  adaptation rules       cost + call log       Alembic
  scheduling             embeddings/retrieval  Redis (cache + arq queue)
```

Workers (**arq** — async-native, Redis-only; Celery is overkill here) handle: resource URL validation, roadmap generation, embedding backfill, adaptation recompute, eval runs.

### Repo layout
```
pathwise/
├─ docker-compose.yml, .env.example, README.md, Makefile
├─ .github/workflows/ci.yml
├─ backend/
│  ├─ pathwise/
│  │  ├─ api/routes/{auth,onboarding,roadmap,concepts,resources,
│  │  │              assessments,tutor,projects,dashboard,admin}.py
│  │  ├─ api/{deps.py,errors.py,schemas/}
│  │  ├─ services/{roadmap,recommendation,assessment,tutor,
│  │  │            resource,knowledge,decision,adaptation,evaluation}/
│  │  ├─ ai/
│  │  │  ├─ providers/{base,anthropic_provider,openai_provider,fake_provider}.py
│  │  │  ├─ prompts/{registry.py, <name>/v1.md ...}
│  │  │  ├─ {embeddings,retrieval,validators,cache,cost,call_log}.py
│  │  ├─ models/            # SQLAlchemy ORM
│  │  ├─ database/          # session, base, migrations/ (alembic)
│  │  ├─ workers/           # arq tasks
│  │  ├─ evaluation/        # runner + scorers
│  │  ├─ data/{knowledge_graph,resources}/*.yaml   # curated seeds
│  │  └─ utils/
│  ├─ evals/datasets/*.jsonl
│  └─ tests/{unit,integration,e2e}
└─ frontend/
   ├─ src/{components,pages,features/{roadmap,dashboard,tutor,
   │        assessments,resources,projects,onboarding},hooks,services,types,utils}
   └─ tests/, e2e/ (Playwright)
```

---

## 2. Database Schema

**Identity & profile** — `users` (id, email, password_hash argon2, created_at), `refresh_tokens` (hashed, rotating), `learning_profiles` (goal_text, goal_concept_ids, experience_summary, learning_style enum, hours_per_week, deadline, timezone).

**Knowledge graph (global, shared across users)**
- `concepts` — id, slug (unique), name, domain, description, difficulty 1–5, estimated_minutes, `learning_objectives` JSONB, tags, `embedding vector(384)`, source (`seed`|`llm`), status (`approved`|`pending`)
- `concept_edges` — src_id, dst_id, `relation` enum (`PREREQUISITE_OF`, `DEPENDS_ON`, `RELATED_TO`, `BUILDS_ON`, `ALTERNATIVE_TO`), strength 0–1, source, **unique(src,dst,relation)**

**Roadmap (per user, versioned append-only)**
- `roadmaps` — user_id, title, goal, status, current_revision
- `roadmap_nodes` — roadmap_id, concept_id, order_index, node_type (`topic`|`practice`|`assessment`|`project`|`review`), status (`not_started`|`in_progress`|`completed`|`needs_review`|`locked`|`recommended`), estimated_minutes, rationale
- `roadmap_edges` — prerequisite links *within* a roadmap (a filtered projection of the global graph)
- `roadmap_revisions` — revision_no, `mutations` JSONB, `trigger` JSONB (the evidence), `explanation` text, created_at ← **this table is the "why did my roadmap change" feature**

**Knowledge state**
- `mastery_states` — user_id, concept_id, alpha, beta (Beta-distribution pseudo-counts), mastery, confidence, evidence_count, last_evidence_at, review_due_at
- `evidence_events` — user_id, concept_id, source enum (`assessment`|`quiz`|`project`|`lesson`|`self_report`|`tutor`|`time_on_task`), raw_score, weight, occurred_at, payload JSONB — **append-only; mastery is always recomputable from this log**

**Resources** — `resources` (title, url unique-canonical, resource_type, difficulty, duration_minutes, publisher, published_at, quality_prior, last_validated_at, http_status, `embedding vector(384)`), `resource_concepts` (relevance 0–1), `resource_chunks` (text + embedding, for tutor RAG)

**Assessments** — `assessments`, `questions` (type, stem, options JSONB, answer_key, `concept_ids[]`, `objective_ids[]`, difficulty), `attempts`, `answers` (response, score, `objective_scores` JSONB, `misconceptions` JSONB)

**Projects** — `projects` (objective, requirements, skills[], difficulty, expected_hours, rubric JSONB, extensions), `project_submissions`

**Tutor** — `tutor_sessions`, `tutor_messages` (role, content, tool_calls JSONB, retrieved_context JSONB)

**AI observability** — `llm_calls` (feature, prompt_name, prompt_version, prompt_hash, model, input/output/cache tokens, cost_usd, latency_ms, status, request_id, validation_passed, retry_count), `eval_runs`, `eval_results`

Indexes: HNSW on every `vector(384)` column; composite `(user_id, concept_id)` on mastery/evidence; GIN on `concept_ids[]`.

---

## 3. Deterministic Core (the heart — build first, no LLM)

### 3.1 Graph algorithms — `services/knowledge/graph.py`
Pure functions over an in-memory adjacency snapshot (Redis-cached, invalidated on graph write):
- `prerequisite_closure(concept, max_depth)` — transitive ancestors
- `topological_order(subgraph)` — Kahn's algorithm; **rejects cycles**
- `unlocked_frontier(user, roadmap)` — nodes whose prereqs are satisfied
- `blame_candidates(concept, mastery_map)` → ranked prereqs, scored
  `score = mastery_deficit(p) × edge_strength × decay^hops` — this is the "which prerequisite is causing the difficulty" answer, and it is arithmetic, not a prompt.

**Cycle prevention is an invariant:** every write of a `PREREQUISITE_OF`/`DEPENDS_ON` edge (seed, LLM-proposed, or user) runs a DAG check and is rejected on failure.

### 3.2 Mastery model — `services/knowledge/mastery.py`
Mastery is `Beta(α, β)`; each evidence event adds pseudo-counts scaled by a source-reliability weight (project 1.2, assessment 1.0, quiz 0.8, tutor-signal 0.2, lesson 0.25, self-report 0.15). Then:
- **mastery** = `α / (α+β)`; **confidence** = `1 − Var(Beta)` normalized — a 90% from one quiz is *not* the same as 90% from twelve events, and the UI shows that.
- **Upward propagation:** success on C credits its prerequisites at damping `γ=0.35` per hop, ≤2 hops (this is what lets Pathwise *skip* material — spec §2).
- **Forgetting curve:** `m_eff = m · exp(−Δt / τ)` with `τ` growing in mastery and review count → drives `needs_review` and spaced repetition.
- Failure does **not** propagate downward as mastery; it triggers `blame_candidates` instead.

All pure, all unit-tested including property tests (monotonicity, bounds, order-independence of same-timestamp events).

### 3.3 Decision engine — `services/decision/` (**no LLM**)
1. **Candidates:** roadmap nodes with `m_eff < 0.75` whose prereqs clear `0.6 × edge_strength`, plus review-due nodes.
2. **Score:** weighted sum of goal_relevance (graph distance to goal concepts), readiness (prereq margin), urgency (deadline vs. remaining hours), review_debt, difficulty_fit (penalty on |difficulty − learner level|), remediation_boost, attempt_penalty.
3. **Emit** `DecisionTrace` — every term, weight, and contribution, as data.

The LLM's only job is turning that trace into the paragraph. A `TraceGroundedValidator` then checks the prose cites the top-weighted factor and asserts no number absent from the trace — reject and retry once, else fall back to a deterministic template. **This is the core hallucination-mitigation story.**

### 3.4 Adaptation engine — `services/adaptation/`
Event-driven (assessment graded, project graded, N tutor struggle signals). Rules emit typed `RoadmapMutation`s:
`INSERT_REMEDIATION` (blame-driven), `COMPRESS`/`SKIP` (mastery ≥ 0.85 on diagnostic), `ADD_PRACTICE`, `ADD_REVIEW`, `REORDER`, `SPLIT_NODE`.
Each writes a `roadmap_revisions` row carrying the triggering evidence and the explanation — reproducing the spec's exact example: *"You scored 48% on the gradient descent assessment. Your calculus performance is strong, so I've added a short optimization section."*

---

## 4. AI Layer — `backend/pathwise/ai/`

**Provider abstraction** (`providers/base.py`): `complete()`, `complete_structured(schema)`, `stream()`, `count_tokens()`. Three implementations: Anthropic (primary), OpenAI (adapter), **Fake** (fixture-driven, deterministic — every test and all of CI runs on it).

**Anthropic specifics** (verified against the bundled `claude-api` skill, not from memory):
- Model `claude-opus-5`; `thinking={"type":"adaptive"}`; depth via `output_config={"effort": ...}` — `low` for classification/labeling, `high` for roadmap/tutor. **No `budget_tokens`** (400s on Opus 5).
- **Structured output** via `client.messages.parse(..., output_format=PydanticModel)` → `response.parsed_output`. Not prefill, not "respond in JSON" prompting.
- Tool calling with `strict: True` + `additionalProperties: false` for the tutor's tools.
- `cache_control={"type":"ephemeral"}` on the stable system prefix; **assert `usage.cache_read_input_tokens > 0`** in an integration test so a silent cache invalidator gets caught.
- Streaming for the tutor; typed-exception chain (`NotFoundError` → `RateLimitError` → `APIStatusError` → `APIConnectionError`), SDK `max_retries` for transport, our own retry only for validation failures.
- `stop_reason == "refusal"` handled explicitly.

**Prompt versioning:** prompts are files (`prompts/roadmap_generate/v1.md`); a registry maps name → active version; every call records name + version + content hash. Changing a prompt is a diff, and the eval suite tells you if it regressed.

**Validation pipeline** — every structured call: Pydantic parse → domain validators (concept slugs exist, prerequisite edges form a DAG, **resource URLs exist in the catalog**, question has exactly one correct MCQ answer) → on failure one repair round-trip with the error text → on second failure raise and take the deterministic fallback path. Nothing unvalidated reaches the database.

**Cost tracking:** pricing table keyed by model id (`claude-opus-5` = $5/$25 per MTok), applied to `usage` incl. cache read/creation, written to `llm_calls` per call. Dashboard endpoint aggregates per user/feature.

**Retrieval:** fastembed 384-dim embeddings → pgvector HNSW. Used for resource recommendation rerank, tutor RAG over `resource_chunks`, and concept dedupe when the LLM proposes new graph nodes.

---

## 5. Resource Pipeline (spec §4 — no invented URLs)

Curated YAML (`data/resources/*.yaml`) — ~200–300 real, well-known resources: official docs, MIT/Stanford/fast.ai courses, 3Blue1Brown, canonical papers and books, interactive sites. Ingest CLI runs: schema validate → canonicalize URL → **HTTP check (follow redirects, record final URL + status)** → dedupe → publisher quality prior → embed → store, with `resource_concepts` relevance.

Recommendation is a three-stage funnel: deterministic prefilter (concept, difficulty band from current mastery, duration ≤ remaining weekly budget, learning-style type weighting) → pgvector semantic rerank → LLM writes the *why* using **only fields from the retrieved row**. A background worker re-validates URLs weekly and flags dead links.

---

## 6. API Surface (representative)

```
POST /auth/{register,login,refresh,logout}       GET  /me
POST /onboarding/goal            → parsed goal + target skill profile
POST /onboarding/diagnostic      → generate;  POST /diagnostic/{id}/submit
POST /roadmaps                   → generate (async job + SSE progress)
GET  /roadmaps/{id}              → nodes + edges + statuses (React Flow shape)
GET  /roadmaps/{id}/revisions    → adaptation history + explanations
POST /roadmaps/{id}/nodes/{n}/complete
GET  /decisions/next             → recommendation + DecisionTrace + explanation
GET  /concepts/{id}              → concept, objectives, prereqs, resources, mastery
GET  /resources?concept=&style=  → ranked + per-resource rationale
POST /assessments/generate       POST /assessments/{id}/submit
POST /tutor/sessions             POST /tutor/sessions/{id}/messages  (SSE stream)
GET  /dashboard                  → progress, mastery-by-subject, weak areas, next action
GET  /admin/ai/usage             → tokens, cost, latency, validation-failure rate
```

---

## 7. Frontend

React 19 + TS + Vite + Tailwind + **React Flow** + TanStack Query + react-router. Recharts for mastery/progress charts.

- **Roadmap canvas:** custom React Flow nodes with the six visual states (not started / in progress / completed / needs review / locked / recommended next), mastery ring, difficulty and time chips; zoom/pan/minimap; click → side panel with objectives, resources, assessments, prerequisite trail. Dagre for layered layout.
- **Dashboard:** the single most prominent element is **"What should I do next?"** — the decision-engine recommendation with its reason and estimated time, one click to start.
- **Tutor:** streaming chat with a visible learner-context strip (what the tutor knows about you) and inline prerequisite links.
- I'll load the `frontend-design` skill before building the UI so it reads as a product, not a dev dashboard. Accessible components, real loading/error/empty states, responsive.

---

## 8. Evaluation Framework (spec §12)

Datasets as JSONL in `backend/evals/datasets/`. Scorers, deterministic first:
- **schema validity rate**, **prereq-DAG validity**, **resource URL liveness**, **concept-coverage recall vs. gold roadmaps**, **decision-engine top-1 agreement** with hand-labeled cases, **blame-attribution accuracy** on synthetic mastery profiles.
- LLM-as-judge (rubric, structured output, run at high effort) only for tutor-response and roadmap quality, with a calibration subset to keep the judge honest.

Runner: `python -m pathwise.evaluation.run --suite roadmap` → writes `eval_runs`/`eval_results`, prints a table, **exits nonzero on regression against the stored baseline**. Records prompt version, model, latency, tokens, cost per case.

---

## 9. Implementation Phases

Each phase ends with tests green and a working slice; I stop and show you before moving on.

| # | Phase | Delivers |
|---|---|---|
| 0 | Scaffold | docker-compose (pg+pgvector, redis, api, worker), env config, Alembic, CI, README, Makefile |
| 1 | **Deterministic core** | Full schema + migrations, auth (JWT+argon2+rotation), seed knowledge graph (ML/CS/math domains), graph algorithms, mastery math — **heavily unit-tested, zero LLM** |
| 2 | AI layer | Provider interface + Anthropic + fake, prompt registry, cost/call logging, validation+repair, embeddings, pgvector retrieval |
| 3 | Onboarding → roadmap | Goal parsing, diagnostic generation/grading, roadmap generation w/ graph grounding, versioned persistence |
| 4 | Resources | Curated catalog, ingest+validation CLI, recommendation funnel + rationale |
| 5 | Adaptive engine | Evidence events, mastery updates, decision engine + trace, adaptation rules + revision history |
| 6 | Assessments | MCQ + short-answer + conceptual, rubric grading, misconception extraction → evidence |
| 7 | Frontend I | Auth, onboarding, React Flow roadmap, node detail, dashboard w/ "what's next" |
| 8 | Tutor | Tool-calling + RAG tutor, SSE streaming, frontend chat → **MVP complete** |
| 9 | Evaluation | Datasets, scorers, runner, CI regression gate, AI usage/cost dashboard |
| 10 | Advanced | Projects + rubric grading, sandboxed coding assessments, spaced repetition, resource quality scoring, deploy configs (Vercel + Fly/Railway + managed PG) |

---

## 10. Verification

- **Unit:** pytest over graph algorithms, mastery math, decision scoring, adaptation rules, validators. Property-based tests (Hypothesis) for mastery bounds/monotonicity and DAG invariants.
- **Integration:** pytest + `testcontainers`-style ephemeral Postgres — migrations, repositories, pgvector search, full onboarding→roadmap→assessment→adaptation flow against the **fake LLM provider** (deterministic, offline, free).
- **Contract:** a small suite that runs against the **real** Anthropic API (marked `@pytest.mark.live`, opt-in) asserting structured-output parsing, tool calling, and a nonzero `cache_read_input_tokens`.
- **Frontend:** Vitest + Testing Library for components/hooks; **Playwright** e2e for register → onboard → diagnostic → roadmap → complete node → see roadmap adapt.
- **Evals:** `make eval` runs all suites and fails on regression.
- **Manual:** `docker compose up` → seeded demo user with a pre-populated roadmap so the adaptive behavior is visible immediately, without waiting through a full study cycle.

**Prerequisite before anything runs:** Docker Desktop installed, and `ANTHROPIC_API_KEY` in `backend/.env` (phases 0–1 need neither; phase 2 onward needs both).

---

## Open risk I want to flag

The knowledge graph's quality ceilings everything downstream — a bad prerequisite edge produces a bad roadmap, bad blame attribution, and a bad next-topic decision. So I'm hand-authoring the seed graph for ML/CS/math (the deepest domain) rather than generating it, and gating any LLM-proposed concept or edge behind DAG validation, embedding-based dedupe, and a `pending` status. Other domains (cybersecurity, data science, web) get shallower seeds initially and grow through the same validated pipeline.
