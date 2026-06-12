# AI Model Pipeline

AI model component for a Pakistani hospital's Clinical Decision Support
System (CDSS). Takes patient vitals + chief complaint as input and
returns a diagnosis + treatment + medicines + citations JSON, with
mandatory physician sign-off. This repo covers **only the model** — a
separate team integrates it into the hospital management web app via
the included REST API.

**This model is decision-support only.** Every output requires
physician sign-off before any action is taken on it.

---

## 1. Pipeline overview

```
Patient vitals + chief complaint
        |
        v
[1] rule_engine.py        --> critical_alert (runs BEFORE any AI)
        |
        v
[2] XGBoost classifier     --> probability distribution over conditions
    (classifier.pkl + label_encoder.pkl, structured vitals only)
        |
        v
[3] build_rag_index.py      --> ChromaDB index over knowledge_base.json
    (build once, ahead of time)
        |
        v
[4] llm_pipeline.py         --> RAG retrieval + Mistral 7B (via Ollama)
    --> raw diagnosis/medicines JSON proposal
        |
        v
[5] citation_validator.py   --> citation enforcement, confidence gating,
                                 drug safety overrides (dengue/NSAID,
                                 TB referral, conditional antibiotics)
        |
        v
   Final JSON output (main.py / api_server.py)
```

## 2. Repo structure

```
cdss_phase2/
├── api_server.py          <- HTTP API (FastAPI) - call this from the web app
├── main.py                 <- pipeline orchestration (called by api_server.py)
├── rule_engine.py           <- critical alert logic
├── knowledge_base.json      <- WHO-cited medicine/condition database (15 conditions)
├── build_rag_index.py       <- builds the RAG search index (run once)
├── llm_pipeline.py           <- LLM (Mistral) reasoning layer
├── citation_validator.py     <- safety gate (citations, confidence, drug rules)
├── requirements.txt
├── test_requests.json        <- 6 ready-made example patients for testing
├── README.md                  <- this file
└── data/
    ├── classifier.pkl
    └── label_encoder.pkl
```

| File | Purpose |
|---|---|
| `rule_engine.py` | Pure-Python critical alert logic (BP, blood sugar, temp, pulse, BMI thresholds). Runs first, independent of ML/LLM. |
| `knowledge_base.json` | Starter KB: 15 conditions with real WHO EML 24th List (2025) citations, adopted as Pakistan's NEML (Gazette 20-Oct-2025). |
| `build_rag_index.py` | Builds a local ChromaDB vector index over the KB using `all-MiniLM-L6-v2` embeddings (one chunk per condition overview + one per medicine). |
| `llm_pipeline.py` | Prompt template, RAG retrieval, Mistral 7B call via Ollama, JSON parsing/repair. |
| `citation_validator.py` | **Safety gate.** Drops any medicine without a KB citation, applies confidence-based gating (>80% full, 60-80% diagnosis only, <60% refer), and hard-coded clinical safety overrides. |
| `main.py` | End-to-end orchestration. Can be run directly via CLI. |
| `api_server.py` | FastAPI wrapper exposing `main.run_pipeline()` as a REST endpoint for the web app team. |
| `data/classifier.pkl`, `data/label_encoder.pkl` | XGBoost classifier + label encoder from Phase 1 training. |

---

## 3. One-time setup (per machine)

Because the LLM runs 100% locally/offline, **each machine that runs
this needs its own full setup** — there's no shared server.

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Ollama (https://ollama.com) and pull the model
ollama pull mistral
ollama serve   # usually runs automatically as a background service

# 3. Build the RAG search index (run once, or whenever
#    knowledge_base.json changes; downloads a small ~80MB
#    embedding model on first run)
python build_rag_index.py
```

Place `classifier.pkl` and `label_encoder.pkl` (from the Day3-5 Colab
script) into `data/`. If absent, the pipeline still runs — the
classifier signal is simply omitted and the LLM prompt notes "no
classifier output available".

---

## 4. Running via CLI (for development/debugging)

```bash
# Run built-in sample patients (malaria/dengue, hypertensive crisis, T2DM)
python main.py

# Run a single patient from a JSON file
python main.py patient.json
```

`patient.json` should contain the same fields as `VitalsInput`:

```json
{
  "age": 32,
  "sex": "F",
  "bp_systolic": 110,
  "bp_diastolic": 72,
  "blood_sugar": 98,
  "blood_sugar_context": "random",
  "weight_kg": 58,
  "height_cm": 162,
  "temperature_c": 39.2,
  "pulse_bpm": 102,
  "chief_complaint": "Fever with chills and sweating for 3 days, severe headache"
}
```

---

## 5. Running via API (for the web app team)

Start the server:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` — an interactive Swagger UI
where you can test requests directly without writing any code.

### `POST /diagnose`

Send patient vitals, get back the full decision-support JSON.

**Request body example:**

```json
{
  "age": 32,
  "sex": "F",
  "bp_systolic": 110,
  "bp_diastolic": 72,
  "blood_sugar": 98,
  "blood_sugar_context": "random",
  "weight_kg": 58,
  "height_cm": 162,
  "temperature_c": 39.2,
  "pulse_bpm": 102,
  "chief_complaint": "Fever with chills and sweating for 3 days, severe headache, body aches"
}
```

All fields are optional except `chief_complaint`, but a useful
diagnosis needs all vitals filled in. `sex` must be `"M"` or `"F"`.
`blood_sugar_context` is `"fasting"`, `"random"`, or `"post_meal"`.

**curl example:**

```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "age": 60, "sex": "M",
    "bp_systolic": 188, "bp_diastolic": 124,
    "blood_sugar": 110, "blood_sugar_context": "random",
    "weight_kg": 82, "height_cm": 172,
    "temperature_c": 36.9, "pulse_bpm": 90,
    "chief_complaint": "Severe headache and blurred vision"
  }'
```

**Response shape (key fields):**

| Field | Meaning |
|---|---|
| `critical_alert` | Emergency flag from the rule engine. `flag: true` means the UI should show this immediately, regardless of the rest of the response. |
| `diagnosis` | `{name, icd10}` — empty `{}` if referred to specialist. |
| `confidence` | 0.0-1.0. Below 0.6 → refer; 0.6-0.8 → diagnosis only; above 0.8 → full recommendation. |
| `differential_diagnoses` | Other possibilities the model considered. |
| `medicines` | Array of `{name, dosage_form, dose_instruction, tier, citation, drap_check, flag?}`. Empty if `refer_to_specialist` is true. |
| `treatment_plan_notes` | Non-medicine advice (lifestyle, etc.) |
| `action` | `"full_recommendation"` \| `"diagnosis_only"` \| `"refer"` |
| `refer_to_specialist` | `true`/`false` |
| `refer_reason` | Human-readable reason if referred. |
| `classifier_output` | Top-5 XGBoost predictions (for transparency/audit, not for display as "the answer"). |
| `llm_reasoning` | 2-3 sentence plain-text explanation from the LLM. |

### `GET /health`

Quick check that the server started correctly:
```json
{ "status": "ok", "classifier_loaded": true, "kb_conditions_loaded": 15 }
```

### `GET /conditions`

Lists the 15 conditions currently in the knowledge base, with their
ICD-10 codes — useful for understanding current model coverage.

---

## 6. CRITICAL: Response time

**Each `/diagnose` call takes roughly 1-4 minutes** on CPU-only
hardware (no GPU). This is because of local LLM inference (Mistral
7B via Ollama) — it is NOT a network or server bug.

**Implications for the web app:**
- Use a request timeout of **at least 300 seconds (5 minutes)**.
- Show a loading/spinner state — do NOT treat this like a typical
  sub-second API.
- Consider calling `/diagnose` asynchronously (e.g. submit + poll, or
  websocket/job-queue pattern) if your UI framework times out on long
  HTTP requests by default.
- The first request after starting the server may be even slower
  (model warm-up).

---

## 7. Testing

Use `test_requests.json` — it contains 6 ready-made example patients,
each labeled with the behavior it's designed to demonstrate:

1. **Hypertensive crisis** — instant emergency alert + full medicine recommendation
2. **Severe hypoglycaemia** — instant emergency alert
3. **Vague/borderline symptoms** — low confidence → refer to specialist, no medicines
4. **Fever/dengue case** — tests the NSAID safety block
5. **Paediatric patient (age 6)** — adult thresholds shouldn't apply, special warning
6. **Diabetes symptoms** — shows the classifier's known bias

Copy any `request` object from that file and paste it into the
`/diagnose` endpoint in Swagger UI (`http://localhost:8000/docs`).

---

## 8. Safety mechanisms implemented (Phase 2)

- **Citation enforcement**: any medicine without a matching KB entry +
  citation is dropped; if the diagnosis itself has no KB grounding, the
  case is referred to a specialist with no medicines shown.
- **Confidence gating**: >80% full recommendation, 60-80% diagnosis only
  ("verify with specialist"), <60% refer entirely (no medicines).
- **Dengue/NSAID hard block**: if dengue is in the differential list,
  ibuprofen/aspirin/diclofenac/naproxen/mefenamic acid are blocked
  regardless of confidence.
- **TB referral-only**: `TB_PULM` always routes to "refer to national TB
  programme / DOTS"; its medicine entry is reference-only and never
  shown to the patient/physician as an active prescription.
- **Conditional antibiotics**: PUD's H. pylori eradication regimen and
  gastroenteritis's ciprofloxacin are only shown if the chief complaint
  documents the relevant clinical trigger (H. pylori positivity /
  dysentery features).
- **URI antibiotic review flag**: phenoxymethylpenicillin/amoxicillin for
  pharyngitis are flagged for explicit bacterial-vs-viral physician
  review rather than auto-recommended.
- **Critical alert priority**: `rule_engine.py` output is always attached
  to the final JSON as `critical_alert`, independent of AI confidence —
  the hospital UI should surface this immediately regardless of
  diagnosis.

---

## 9. Known limitations (Phase 2)

- DRAP National Drug List check is a placeholder (`check_drap_registration`
  returns `"not_checked"`); KB medicines are currently treated as
  PNF/DRAP-aligned because Pakistan's NEML 2025 adopted the WHO 24th
  list verbatim (Gazette 20-Oct-2025).
- Classifier accuracy is currently ~65% overall with severe class
  imbalance (see `training_data_card.json`); it is used as a weak
  signal only, not ground truth.
- Vitals input does not capture SpO2 or respiratory rate, which limits
  asthma/CAP severity assessment (documented in `knowledge_base.json`
  `confidence_notes`).
- `sex` encoding in `main.py`'s classifier wrapper (`M=1, F=0`) should
  be verified against the actual encoding used in the Day3-5 training
  script.
- **15 conditions covered** in the current knowledge base. Anything
  outside this scope correctly results in `refer_to_specialist: true`.

---

## 10. Next steps

- Replace placeholder/Tier4 posology with PNF-specific dosing tables.
- Wire `check_drap_registration()` to a live DRAP National Drug List
  source.
- Level 1/2/3 validation per the project validation plan (50 hand-crafted
  cases, 100-case doctor review, MedQA/MedMCQA benchmark).
