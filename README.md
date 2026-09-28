# US4AI

US4AI is a research prototype for deriving risk-informed requirements for AI-enabled systems using the NIST AI RMF Playbook. It supports inspectable evidence, contextual risk inference, and requirement derivation. It does not claim empirical effectiveness or replace researcher and stakeholder review.

## Conceptual pipeline

System Purpose + User Story + optional Acceptance Criteria + AI Tasks
→ NIST evidence retrieval → contextual AI risk inference
→ AI-specific requirement derivation → validated references and traceability.

These are three distinct artifacts:

- **NIST evidence:** retrieved source guidance.
- **Contextual AI risk:** a model-generated inference about the supplied scenario, grounded in that guidance. It is not a direct NIST risk statement.
- **AI-specific requirement:** an actionable derivation addressing contextual risks, justified by the scenario and NIST evidence. It specifies an implementable system-level control, constraint, mechanism, or behavior; organizational recommendations alone are not requirements.

Reference validation is deterministic; semantic correctness and scientific quality still require review.

## Input

The UI constructs the existing `US4AIAnalysisInput` contract:

- Exactly one nonblank System Purpose.
- Exactly one nonblank User Story.
- Zero or more Acceptance Criteria, entered one per line.
- One or more AI Tasks, each with a nonblank `category` and `task`.

The study UI uses the exact version-controlled vocabulary in `src/ai_task_catalog.py`. Add rows containing dependent Group and AI Task dropdowns; duplicate pairs and incomplete selections are rejected. The backend `AITask` schema remains open to programmatic category/task strings. Task references (`TASK-01`, `TASK-02`, etc.) follow input order within each analysis. Reordering tasks changes these references.

## Knowledge and retrieval

The sole retrieval source is `catalogs/nist_ai_rmf_playbook.json`: 72 unchanged NIST Playbook records. One subcategory is one vector document, identified by its original title.

Retrieval text includes labeled title, description, about, suggested actions, and topics. Original documentation, references, and actor fields remain available through full-record resolution; the source is not rewritten.

The query deterministically includes all scenario fields. The existing defaults retrieve 15 vector candidates and rerank to five evidence items:

- Embedding: FastEmbed `BAAI/bge-small-en-v1.5`, maximum length 512, document mode `default`.
- Reranker: `BAAI/bge-reranker-base`.
- Explicit Chroma collection: `us4ai_nist_playbook`.
- Default persistence: project-relative `data/chroma_nist/`, ignored by Git.

Structured `RetrievedNistEvidence` preserves title, function, category, retrieval text, vector distance, and reranking score. Lower vector distance is better; reranking scores are raw relative-ordering scores, not probabilities. Neither score measures risk severity or requirement importance.

## Generation and traceability

LangGraph runs `retrieve_nist_evidence` → `infer_contextual_risks` → `derive_ai_requirements`. Generation uses OpenAI + Instructor with Pydantic outputs, `gpt-4o-mini`, temperature 0, Instructor `max_retries=1`, and SDK `max_retries=0`.

Prompts distinguish SOURCE, INFERENCE, and DERIVATION. They provide the complete scenario and selected substantive fields of the retrieved source records, including documentation questions. Reference bibliographies are not treated as guidance, and repeated retrieval text is omitted from generation context. Prompts prohibit fabricated quotations and unsupported implementation details. Retrieval relevance alone does not justify a risk: descriptions must explain concrete scenario-to-harm connections, and retrieved evidence may remain unused. Requirements translate supported risks into system-level controls rather than paraphrasing Suggested Actions.

Deterministic validation rejects duplicate risk/requirement IDs, empty required references, unknown tasks, unknown evidence, and unknown addressed risks. It does not silently repair invalid references. Duplicate references and requirement task links outside the addressed risks are rejected. A narrow statement-subject check rejects explicit organization/project-team/stakeholder/management/developer recommendations; it is not a semantic proof of implementability. The traceability table contains only explicit edges from the validated outputs: task/risk, evidence/risk, risk/requirement, task/requirement, and evidence/requirement. It does not invent direct task/evidence relationships or combinations of independent references.

The graph distinguishes no evidence, no sufficiently supported risks, and no justified requirements. None establishes that the scenario is risk-free. The UI preserves validated earlier stages if a later stage fails.

## Environment and installation

The recorded environment uses Python 3.12.14 (`.python-version`). Existing pinned versions are in `requirements-lock.txt`; `requirements.txt` records the broader dependency declarations. No broad upgrades are required for this integration.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

Configure `OPENAI_API_KEY` through the environment or the project-root `.env` file. Never commit that file or share the key. Optional configuration:

```text
NIST_CHROMA_PATH=data/chroma_nist
```

Relative index paths resolve against the project root, independently of the shell directory. The default path is ignored; if an alternative path is used, keep its generated contents out of Git as well. Configuration changes require restarting the app. The application never exports keys or environment contents.

The store factory preserves the `pysqlite3` compatibility substitution for systems whose SQLite is older than Chroma requires. Embeddings and reranking run locally; first use may need model downloads. Generation sends the scenario and selected retrieved NIST content to the configured OpenAI service. Use appropriate research inputs.

## Index initialization and validation

```bash
python -B -m src.nist_index
```

The index validates exactly 72 identities, stored text and metadata, and a corpus/embedding-configuration fingerprint. A valid index is reused without vector writes. Mismatches fail clearly rather than silently replacing data.

Streamlit builds the index when its configured directory is absent. An existing directory is opened and validated without rebuilding. If initialization fails, inspect the configuration and run the validation command. Back up incompatible data and explicitly choose a fresh path or manage the old index outside the app. The UI never offers an automatic destructive rebuild.

Index/model resources are cached across Streamlit reruns. Generation results are not globally cached; each session retains only its own submitted analysis. Restart the app after intentionally changing source/index/configuration files.

## Run the application

```bash
streamlit run app.py
```

Enter the complete scenario and submit **Run analysis**. The four result tabs show source evidence, contextual risks, requirements, and traceability. The displayed input is the submitted scenario, even if the form is subsequently edited.

Download the JSON analysis for later empirical evaluation. It contains the input, task-reference mapping, evidence and resolved source records, inferred risks, derived requirements, explicit traceability, status, and allowlisted model/source metadata. An incomplete analysis is marked as such. A separate CSV contains the traceability edges. No database or environment secrets are exported.

The inactive legacy optional input scanner is not part of this UI. The current conservative prompts and deterministic reference checks are not a comprehensive prompt-injection defense.

## Tests

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

The offline suite uses controlled model/retrieval substitutes. It covers original-source preservation, input contracts, index/evidence contracts, query construction, generation/reference validation, graph routing, input conversion, traceability/export, partial results, and basic Streamlit integration. It does not require real OpenAI calls or model downloads.

A real end-to-end smoke test is a separate operation requiring an available key and local index. One successful run demonstrates execution only, not scientific validity.

## Current limitations

- Long documents and queries are subject to model token limits; one-record-per-document representations are not chunked.
- Retrieval parameters and models are fixed defaults, not empirically tuned here.
- Generated risks and requirements may be incomplete or semantically unsupported despite valid IDs; human review remains necessary.
- IDs are stable within an analysis, not guaranteed identical across different generations.
- The prototype has no authentication, durable analysis storage, or multi-user deployment hardening.
- Local model startup and CPU reranking may be slow or memory intensive.
- The installed `langchain-community` embedding wrapper emits a deprecation warning; the working integration is retained.
- Dependency declarations and the pinned environment are retained without a broad dependency-pruning exercise.
