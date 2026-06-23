# CDSS Phase 2 — AI Model Pipeline

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
[4] llm_pipeline.py         --> RAG retrieval + LLM reasoning
    THREE BACKENDS: Groq (1-2s), Ollama Mistral (200s), MedGemma (20-50s)
    Switch backend with LLM_BACKEND setting in llm_pipeline.py
        |
        v
[5] citation_validator.py   --> citation enforcement, confidence gating,
                                 drug safety overrides, defensive parsing
        |
        v
   Final JSON output (main.py / api_server.py)
```

---

## 2. Repo structure

```
cdss_phase2/
├── api_server.py          <- HTTP API (FastAPI) - call this from the web app
├── main.py                 <- pipeline orchestration (called by api_server.py)
├── rule_engine.py           <- critical alert logic
├── knowledge_base.json      <- WHO-cited medicine/condition database (15 conditions)
├── build_rag_index.py       <- builds the RAG search index (run once)
├── llm_pipeline.py           <- LLM reasoning layer (Groq or Ollama)
├── citation_validator.py     <- safety gate (citations, confidence, drug rules)
├── requirements.txt
├── test_requests.json        <- ready-made example patients for testing
├── run_patient.py            <- interactive terminal runner (paste JSON, get output)
├── debug_pipeline.py         <- debug runner with full error tracebacks
├── test_groq.py              <- quick Groq connection test
├── README.md                  <- this file
└── data/
    ├── classifier.pkl
    └── label_encoder.pkl
```

| File | Purpose |
|---|---|
| `rule_engine.py` | Pure-Python critical alert logic (BP, blood sugar, temp, pulse, BMI thresholds). Runs first, independent of ML/LLM. |
| `knowledge_base.json` | Starter KB: 15 conditions with real WHO EML 24th List (2025) citations, adopted as Pakistan's NEML (Gazette 20-Oct-2025). |
| `build_rag_index.py` | Builds a local ChromaDB vector index over the KB using `all-MiniLM-L6-v2` embeddings. |
| `llm_pipeline.py` | Prompt template, RAG retrieval, LLM call. **THREE backends**: Groq (cloud, 1-2s), Ollama/Mistral (local, 200s), MedGemma (fine-tuned local, 20-50s). Switch via `LLM_BACKEND` setting. Includes auto-generated reasoning and defensive JSON parsing. |
| `citation_validator.py` | **Safety gate.** Drops any medicine without a KB citation, applies confidence-based gating (>80% full, 60-80% diagnosis only, <60% refer), hard-coded clinical safety overrides, and defensive format normalization for LLM responses. |
| `main.py` | End-to-end orchestration. Entry point for CLI and API server. |
| `api_server.py` | FastAPI wrapper exposing `main.run_pipeline()` as a REST endpoint (optional). |
| `data/classifier.pkl`, `data/label_encoder.pkl` | XGBoost classifier + label encoder from Phase 1 training. |

---

## 3. LLM Backend — Groq (fast), Ollama (local), or MedGemma (fine-tuned local)

The LLM backend is controlled by one line at the top of `llm_pipeline.py`:

```python
LLM_BACKEND = "groq"    # change to "ollama" or "medgemma" for local offline mode
```

### Backend Comparison

| | Groq | Ollama (Mistral) | MedGemma (Fine-tuned) |
|---|---|---|---|
| Speed | **1-2 seconds** | ~200 seconds | **20-50 seconds** |
| Internet required | Yes | No | No |
| Cost | Free tier | Free (local) | Free (local) |
| Data privacy | Sends to Groq servers | Stays on machine | 100% offline, no data leaves |
| Model size | 70B (cloud) | 7B | 4B |
| Training | General | General LLM | Fine-tuned on 450 CDSS clinical cases |
| Best for | Demo, dev, web team testing | Development | **Production (patient data)** |

### MedGemma Setup (NEW)

MedGemma is a locally fine-tuned 4B model trained on 450 CDSS-specific clinical cases.
Requires Ollama with the custom model registered:

```bash
# 1. Install Ollama from https://ollama.com
# 2. Create the model using your Modelfile:
ollama create medgemma-cdss-finetuned -f Modelfile

# 3. Start Ollama:
ollama serve

# 4. Set backend in llm_pipeline.py:
# LLM_BACKEND = "medgemma"

# 5. Run pipeline as normal
python run_patient.py
```

**MedGemma auto-fixes:**
- Auto-generates "reasoning" field if model omits it (confidence-based fallback)
- Handles differential diagnoses in any format (dict, string, or mixed)
- Graceful JSON parsing with safe defaults

### API Key security — IMPORTANT (Groq only)
**Never hardcode your Groq API key in any file.** Always set it as an
environment variable in your terminal session before running anything:

**Windows (PowerShell):**
```powershell
set GROQ_API_KEY=gsk_your_key_here
```

**Mac/Linux:**
```bash
export GROQ_API_KEY=gsk_your_key_here
```

This must be run in the same terminal session you use to start the
server or run scripts. It does not persist between terminal sessions —
set it again each time you open a new terminal.

Get a free key at: https://console.groq.com (no credit card required)

### Alternative: Use a .env file (optional)

Instead of setting environment variables in the terminal each time,
create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

The pipeline automatically loads this file on startup (via `python-dotenv`).
**IMPORTANT:** Add `.env` to `.gitignore` to avoid committing API keys to version control.

---

## 4. One-time setup (per machine)

```bash
# 1. Install Python dependencies (includes FastAPI, Uvicorn, python-dotenv)
pip install -r requirements.txt

# 2. Build the RAG search index (run once, or whenever
#    knowledge_base.json changes; downloads ~80MB embedding
#    model on first run)
python build_rag_index.py

# 3. Choose your LLM backend:

# --- Option A: Groq (fastest, requires internet + API key) ---
# Get a free key at console.groq.com, then set it:
#   Windows: set GROQ_API_KEY=gsk_...
#   Linux/Mac: export GROQ_API_KEY=gsk_...
# OR create a .env file with: GROQ_API_KEY=gsk_...

# --- Option B: Ollama (local Mistral 7B, ~200s per request) ---
ollama pull mistral
ollama serve

# --- Option C: MedGemma (local fine-tuned 4B, ~20-50s, RECOMMENDED for patient data) ---
# 3a. Download the MedGemma model and Modelfile
# 3b. Create the model:
ollama create medgemma-cdss-finetuned -f Modelfile
# 3c. Start Ollama:
ollama serve
# 3d. In llm_pipeline.py, set: LLM_BACKEND = "medgemma"
```

Place `classifier.pkl` and `label_encoder.pkl` into `data/`. If absent,
the pipeline still runs — the classifier signal is simply omitted.

---

## 5. Running locally via terminal (quickest way to test)

Use `run_patient.py` — paste any patient JSON directly in the terminal
and get the full pipeline output without starting a server:

```powershell
# Step 1: Set your Groq API key
set GROQ_API_KEY=gsk_your_key_here

# Step 2: Run the interactive terminal runner
python run_patient.py
```

Paste a patient JSON when prompted (see `test_requests.json` for examples),
then press **Enter twice**. Output prints directly to the terminal.

**Example input to paste:**
```json
{
  "age": 60,
  "sex": "M",
  "bp_systolic": 188,
  "bp_diastolic": 124,
  "blood_sugar": 110,
  "blood_sugar_context": "random",
  "weight_kg": 82,
  "height_cm": 172,
  "temperature_c": 36.9,
  "pulse_bpm": 90,
  "chief_complaint": "Severe headache and blurred vision"
}
```

**Expected output:** Emergency critical alert + hypertension diagnosis +
3 medicines with WHO citations, all in ~1.5 seconds via Groq.

You can also test Groq connectivity on its own before running the full pipeline:
```powershell
python test_groq.py
```

---

## 6. Running via API server (for web app team)

### Start the server

```powershell
# Step 1: Set your Groq API key (required every new terminal session)
set GROQ_API_KEY=gsk_your_key_here

# Step 2: Start the API server
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete.` then open:
```
http://localhost:8000/docs
```

This opens **Swagger UI** — an interactive page where you can test the
API without writing any code.

### Testing via Swagger UI (browser)

1. Open `http://localhost:8000/docs`
2. Click **POST /diagnose** → click **Try it out**
3. Delete the default JSON in the text box
4. Paste any `request` object from `test_requests.json`
5. Click **Execute**
6. Wait ~1-2 seconds (Groq) — response appears under "Server response"

### Testing via terminal (curl alternative on Windows)

Open a **second terminal** (keep the server running in the first), then:

```powershell
python run_patient.py
```

Paste the patient JSON, press Enter twice — gets routed through the
same pipeline as the API.

### API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/diagnose` | POST | Run full pipeline, returns diagnosis JSON |
| `/health` | GET | Check server started correctly |
| `/conditions` | GET | List all 15 KB conditions with ICD-10 codes |
| `/docs` | GET | Interactive Swagger UI for browser testing |

### Response shape (key fields)

| Field | Meaning |
|---|---|
| `critical_alert` | Emergency flag from rule engine. `flag: true` = show immediately in UI regardless of everything else. |
| `diagnosis` | `{name, icd10}` — empty `{}` if referred to specialist. |
| `confidence` | 0.0-1.0. Below 0.6 → refer; 0.6-0.8 → diagnosis only; above 0.8 → full recommendation with medicines. |
| `differential_diagnoses` | Other conditions the model considered. |
| `medicines` | Array of `{name, dosage_form, dose_instruction, tier, citation, drap_check}`. Empty if refer_to_specialist is true. |
| `treatment_plan_notes` | Non-medicine advice (lifestyle, diet, etc.) |
| `action` | `"full_recommendation"` / `"diagnosis_only"` / `"refer"` |
| `refer_to_specialist` | `true`/`false` |
| `refer_reason` | Plain English reason if referred. |
| `classifier_output` | Top-5 XGBoost predictions (audit trail, not the diagnosis). |
| `llm_reasoning` | 2-3 sentence plain English explanation from the LLM. |

---

## 7. Response time

| Backend | Response time | Notes |
|---|---|---|
| Groq (cloud) | **1-2 seconds** | Requires internet + API key |
| MedGemma (local) | **20-50 seconds** | Fully offline, fine-tuned for CDSS, RECOMMENDED for patient data |
| Ollama (local) | ~200 seconds | Fully offline, first call slower (model load) |

**For the web app:** 
- Groq: use a request timeout of at least 30s
- MedGemma: use a timeout of at least 120s
- Ollama: use a timeout of at least 300s

Always show a loading state — never assume instant response.

---

## 8. Test cases

`test_requests.json` contains ready-made patients covering all pipeline behaviors:

| # | Scenario | Expected behavior |
|---|---|---|
| 1 | Hypertensive crisis (BP 188/124) | `critical_alert.flag=true`, emergency, full medicines |
| 2 | Severe hypoglycaemia (sugar 45) | `critical_alert.flag=true`, emergency |
| 3 | Vague symptoms | `confidence < 0.6`, refer, no medicines |
| 4 | Fever + dengue suspected | NSAID/aspirin blocked from medicine list |
| 5 | Child age 6 | Paediatric warning, adult thresholds not applied |
| 6 | Classic T2DM symptoms | Tests classifier bias (likely says hypertension) |

Copy any `request` object from that file and use it in `run_patient.py`
or Swagger UI.

---

## 9. Debugging

If you get errors, run the debug script instead of the API server —
it shows the full error traceback:

```powershell
python debug_pipeline.py
```

This runs the hypertensive crisis sample through the entire pipeline
and prints exactly where and why it fails.

---

## 10. Safety mechanisms (Phase 2)

- **Citation enforcement**: any medicine without a KB citation is dropped.
  If the diagnosis has no KB grounding at all, case is referred to specialist.
- **Confidence gating**: >80% full recommendation, 60-80% diagnosis only, <60% refer entirely.
- **Dengue/NSAID hard block**: ibuprofen/aspirin/diclofenac/naproxen/mefenamic acid
  blocked if dengue is in the differential — regardless of confidence.
- **TB referral-only**: TB_PULM always refers to national TB programme / DOTS.
  Medicine list suppressed entirely.
- **Conditional antibiotics**: PUD H. pylori eradication and gastroenteritis
  (see citation_validator.py for full rules).
- **Defensive JSON parsing**: 
  - Auto-generates "reasoning" field if LLM omits it (uses confidence-based heuristics)
  - Handles `differential_diagnoses` in any format (dicts, strings, or mixed)
  - Graceful fallback to safe defaults if parsing fails
- **Format normalization**: All LLM responses automatically normalized to consistent schema before validation
  ciprofloxacin only shown if clinically documented in chief complaint.
- **URI antibiotic flag**: penicillin/amoxicillin for pharyngitis flagged for
  explicit bacterial-vs-viral physician review.
- **Critical alert priority**: rule_engine output always included in final JSON
  independent of AI confidence — UI must surface this immediately.

---

## 11. Known limitations (Phase 2)

- DRAP National Drug List check is a placeholder (`drap_status: "not_checked"`).
  KB medicines treated as PNF/DRAP-aligned via NEML 2025 = WHO EML 24th List.
- XGBoost classifier ~65% accuracy with severe class imbalance — used as weak
  signal only, not ground truth.
- No SpO2 or respiratory rate input — limits asthma/CAP severity assessment.
- 15 conditions in current KB. Anything outside scope → `refer_to_specialist: true`.
- Groq free tier: 30 requests/minute, 14,400 requests/day limit.

---

## 12. Next steps

- Replace placeholder posology with PNF-specific dosing tables.
- Wire `check_drap_registration()` to live DRAP National Drug List.
- Level 1/2/3 validation (50 hand-crafted cases, 100-case doctor review,
  MedQA/MedMCQA benchmark).