"""
llm_pipeline.py
===============
Phase 2 - Step 4: LLM reasoning layer.

Supports THREE LLM backends — switch with LLM_BACKEND below:

  "groq"      — Groq cloud API (free tier, ~1-2s responses).
                Requires GROQ_API_KEY environment variable.
                NOTE: sends patient data to Groq's servers — dev/demo only.

  "ollama"    — Local Mistral 7B via Ollama (100% offline, ~200s on CPU).
                Requires ollama running locally with mistral model pulled.

  "medgemma"  — Your locally fine-tuned MedGemma 4B model via Ollama.
                Trained on 450 CDSS-specific clinical cases.
                Requires ollama running with medgemma-cdss-finetuned registered.
                Offline, no data leaves the machine, no API limits.
                Run: ollama create medgemma-cdss-finetuned -f Modelfile

Switch between them by changing LLM_BACKEND below.
Everything else (RAG, citation validation, safety gating) is unchanged.
"""
import os
from dotenv import load_dotenv # Add this

# Load environment variables from .env file
load_dotenv() # Add this line
import json
import os
import re
import time
from typing import Dict, List, Optional

import requests
import chromadb
from chromadb.utils import embedding_functions

# =====================================================================
# SWITCH HERE: "groq" | "ollama" | "medgemma"
# =====================================================================
LLM_BACKEND = "groq"   # <-- change to switch backends

# --- Groq settings ---
# NEVER hardcode your API key here. Set as environment variable:
#   Windows: set GROQ_API_KEY=gsk_...
#   Linux/Mac: export GROQ_API_KEY=gsk_...
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

# --- Ollama (Mistral) settings ---
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# --- MedGemma fine-tuned settings ---
# Name must match exactly what you used in: ollama create <name> -f Modelfile
MEDGEMMA_MODEL   = "medgemma-cdss-finetuned"
MEDGEMMA_TIMEOUT = 600   # CPU inference is slow; generous timeout

# --- Shared RAG settings ---
CHROMA_DIR       = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME  = "cdss_kb_phase2"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
N_RAG_RESULTS    = 3   # kept small to stay within token limits


# ---------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------

_collection = None
_embed_fn   = None


def _get_collection():
    """Lazily load the ChromaDB collection + embedding function once."""
    global _collection, _embed_fn
    if _collection is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=_embed_fn
        )
    return _collection


def retrieve_context(query_text: str, n_results: int = N_RAG_RESULTS) -> List[Dict]:
    """
    Retrieve the top-N most relevant KB chunks for the given query.
    Returns a list of dicts: {"text": str, "metadata": dict, "distance": float}
    """
    collection = _get_collection()
    results = collection.query(query_texts=[query_text], n_results=n_results)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})
    return chunks


def _format_context_for_prompt(chunks: List[Dict]) -> str:
    """Render retrieved chunks as a numbered reference list for the prompt."""
    lines = []
    seen_conditions = set()
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        cond_id = meta.get("condition_id", "")
        seen_conditions.add(cond_id)
        lines.append(f"[{i}] (condition_id={cond_id}) {chunk['text']}")
    return "\n".join(lines), sorted(seen_conditions)


# ---------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a clinical decision support assistant for a hospital \
in Pakistan. You are NOT a doctor and your output is decision-SUPPORT only - \
a physician will review and sign off on everything you produce.

You MUST base your diagnosis and medicine choices ONLY on the REFERENCE \
CONTEXT provided below, which comes from the hospital's vetted knowledge \
base (WHO Model List of Essential Medicines, adopted as Pakistan's NEML). \
Do NOT invent conditions, medicines, dosages, or citations that are not \
present in the reference context. If the reference context does not \
adequately cover the patient's presentation, set "condition_id" to null \
and lower your confidence accordingly - the system will refer the case \
to a specialist.

=== PATIENT DATA ===
Age: {age}
Sex: {sex}
Blood pressure: {bp_systolic}/{bp_diastolic} mmHg
Blood sugar: {blood_sugar} mg/dL ({blood_sugar_context})
Weight: {weight_kg} kg, Height: {height_cm} cm (BMI: {bmi})
Temperature: {temperature_c} C
Pulse: {pulse_bpm} bpm
Chief complaint: {chief_complaint}

=== CLASSIFIER OUTPUT (XGBoost, structured-vitals-only model) ===
Top predicted conditions (probability distribution):
{classifier_summary}
NOTE: this classifier was trained on a small, imbalanced dataset and is a \
WEAK SIGNAL ONLY - use it as one input among several, not as ground truth.

=== CRITICAL ALERT (rule engine, runs before AI) ===
{critical_alert_summary}

=== REFERENCE CONTEXT (retrieved from knowledge base - ONLY use these conditions/medicines) ===
{retrieved_context}

=== TASK ===
Based on the patient data, classifier output, and reference context above, \
return a JSON object with EXACTLY this structure and nothing else (no \
markdown, no commentary, no code fences, no Python tuples - use proper \
JSON objects and arrays only):

{{
  "condition_id": "<one of the condition_id values from REFERENCE CONTEXT, or null>",
  "diagnosis": {{"name": "<condition name>", "icd10": "<ICD-10 code>"}},
  "confidence": <float between 0.0 and 1.0>,
  "differential_diagnoses": [
    {{"name": "<name>", "icd10": "<code>"}}
  ],
  "medicines": [
    {{"name": "<medicine name, EXACTLY as it appears in REFERENCE CONTEXT>"}}
  ],
  "reasoning": "<2-3 sentence clinical reasoning, plain text>"
}}

IMPORTANT: You MUST include ALL six fields above in your response.
Do not stop after "diagnosis" — you must also include "confidence",
"differential_diagnoses", "medicines", and "reasoning" every time.

Rules for "medicines": list ONLY medicine names that appear verbatim in the \
REFERENCE CONTEXT for the chosen condition_id. Do not include dosages here - \
those are looked up separately from the knowledge base. If you are not \
confident enough to recommend medicines (confidence < 0.6), return an empty \
medicines list.
"""


def _bmi(weight_kg, height_cm) -> Optional[float]:
    try:
        if weight_kg and height_cm:
            h_m = float(height_cm) / 100.0
            if h_m > 0:
                return round(float(weight_kg) / (h_m * h_m), 1)
    except (TypeError, ValueError):
        pass
    return None


def build_prompt(
    vitals: dict,
    classifier_probs: List[tuple],
    critical_alert: dict,
    retrieved_chunks: List[Dict],
) -> str:
    """
    Build the full prompt string for the LLM.

    vitals: dict with keys matching VitalsInput fields
    classifier_probs: list of (label, probability) tuples, sorted desc
    critical_alert: dict from rule_engine.CriticalAlert.to_dict()
    retrieved_chunks: output of retrieve_context()
    """
    classifier_summary = "\n".join(
        f"  - {label}: {prob:.1%}" for label, prob in classifier_probs[:5]
    ) or "  (no classifier output available)"

    if critical_alert.get("flag"):
        critical_alert_summary = (
            f"FLAGGED - severity: {critical_alert.get('severity')}. "
            f"Reasons: {'; '.join(critical_alert.get('reasons', []))}. "
            f"Recommended action: {critical_alert.get('recommended_action', '')}"
        )
    else:
        critical_alert_summary = "Not flagged - no critical values detected."

    retrieved_context, _ = _format_context_for_prompt(retrieved_chunks)
    if not retrieved_context:
        retrieved_context = "(no relevant reference context retrieved)"

    return PROMPT_TEMPLATE.format(
        age=vitals.get("age", "unknown"),
        sex=vitals.get("sex", "unknown"),
        bp_systolic=vitals.get("bp_systolic", "unknown"),
        bp_diastolic=vitals.get("bp_diastolic", "unknown"),
        blood_sugar=vitals.get("blood_sugar", "unknown"),
        blood_sugar_context=vitals.get("blood_sugar_context", "random"),
        weight_kg=vitals.get("weight_kg", "unknown"),
        height_cm=vitals.get("height_cm", "unknown"),
        bmi=_bmi(vitals.get("weight_kg"), vitals.get("height_cm")) or "unknown",
        temperature_c=vitals.get("temperature_c", "unknown"),
        pulse_bpm=vitals.get("pulse_bpm", "unknown"),
        chief_complaint=vitals.get("chief_complaint", ""),
        classifier_summary=classifier_summary,
        critical_alert_summary=critical_alert_summary,
        retrieved_context=retrieved_context,
    )


# ---------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------

def call_groq(prompt: str) -> str:
    """
    Call the Groq cloud API.
    Requires GROQ_API_KEY set as environment variable.
    Typical response time: 1-2 seconds.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set.\n"
            "  Windows: set GROQ_API_KEY=gsk_...\n"
            "  Linux/Mac: export GROQ_API_KEY=gsk_..."
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a clinical decision support assistant. "
                    "You MUST respond with valid JSON only — no markdown, "
                    "no commentary, no code fences. Just the raw JSON object. "
                    "Always include ALL six fields: condition_id, diagnosis, "
                    "confidence, differential_diagnoses, medicines, reasoning."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }

    start = time.time()
    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    if not response.ok:
        print(f"[llm_pipeline] Groq error {response.status_code}: {response.text}")
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"]
    print(f"[llm_pipeline] Groq ({GROQ_MODEL}) responded in {elapsed:.1f}s")
    return raw_text


def call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 600) -> str:
    """
    Call local Ollama instance with stock Mistral model.
    On i5-8500/12GB RAM CPU-only: expect ~200s per response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": 0.1},
    }
    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    raw_text = _extract_text_from_ollama_response(data)
    print(f"[llm_pipeline] Ollama ({model}) responded in {elapsed:.1f}s; extracted {len(raw_text)} chars")
    if not raw_text:
        print(f"[llm_pipeline] Ollama raw response (truncated): {str(data)[:500]}")
    return raw_text


def call_medgemma(prompt: str) -> str:
    """
    Call your locally fine-tuned MedGemma 4B model via Ollama.

    Key differences from generic Ollama call:
    - format="json" enforced: constrains output to valid JSON syntax,
      preventing the Python-tuple bug seen in raw terminal sessions.
    - Stateless: no conversation history bleeds between calls.
    - keep_alive="60m": keeps model loaded in RAM between pipeline
      calls for faster subsequent responses.
    - System prompt ensures ALL required JSON fields are included.

    Requires:
    - Ollama running: ollama serve
    - Model registered: ollama create medgemma-cdss-finetuned -f Modelfile
    - Modelfile pointing to your downloaded .gguf file

    Typical response time: 20-50s on CPU (i5-8500, 12GB RAM)
    """
    # Prepend system instruction to ensure complete JSON output
    system_instruction = (
        "You are a clinical decision support assistant. "
        "You MUST respond with valid JSON only. "
        "The JSON MUST include ALL six fields: condition_id, diagnosis, confidence, "
        "differential_diagnoses, medicines, and reasoning. "
        "Never omit any field. Always include a concise 2-3 sentence reasoning field."
    )
    
    full_prompt = f"{system_instruction}\n\n{prompt}"
    
    payload = {
        "model": MEDGEMMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "60m",
        "options": {
            "temperature": 0.1,
            "num_predict": 1000,
            "num_ctx": 4096,
        },
    }

    start = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=MEDGEMMA_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "[llm_pipeline] Cannot connect to Ollama. "
            "Make sure Ollama is running: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"[llm_pipeline] MedGemma timed out after {MEDGEMMA_TIMEOUT}s. "
            "Model may be loading for first time — try again in a minute."
        )

    elapsed = time.time() - start
    data = response.json()
    raw_text = _extract_text_from_ollama_response(data)
    print(f"[llm_pipeline] MedGemma ({MEDGEMMA_MODEL}) responded in {elapsed:.1f}s; extracted {len(raw_text)} chars")
    if not raw_text:
        print(f"[llm_pipeline] MedGemma raw response (truncated): {str(data)[:500]}")
    return raw_text


def _extract_text_from_ollama_response(data) -> str:
    """
    Helper to extract a textual response from Ollama-style JSON payloads.
    Ollama responses vary by version/model; try common fields first,
    then recursively search for the largest string value.
    """
    if data is None:
        return ""

    # Common field used earlier
    if isinstance(data, dict):
        if "response" in data and isinstance(data["response"], str):
            return data["response"].strip()
        # some Ollama responses nest choices/results
        if "results" in data and isinstance(data["results"], list) and data["results"]:
            # try to pull content from first result
            first = data["results"][0]
            if isinstance(first, dict):
                # look for 'content' or 'text' or 'response'
                for key in ("content", "text", "response", "output"):
                    if key in first and isinstance(first[key], str):
                        return first[key].strip()
                # deeper: look for message->content
                msg = first.get("message") or first.get("choices")
                if isinstance(msg, dict):
                    for v in msg.values():
                        if isinstance(v, str) and len(v) > 0:
                            return v.strip()
                if isinstance(msg, list) and msg:
                    for item in msg:
                        if isinstance(item, dict):
                            for v in item.values():
                                if isinstance(v, str) and len(v) > 0:
                                    return v.strip()
        # fallback: find largest string value recursively
        largest = ""
        def _search(obj):
            nonlocal largest
            if isinstance(obj, str):
                if len(obj) > len(largest):
                    largest = obj
            elif isinstance(obj, dict):
                for v in obj.values():
                    _search(v)
            elif isinstance(obj, list):
                for v in obj:
                    _search(v)

        _search(data)
        return largest.strip()

    if isinstance(data, list):
        # join string elements or stringify dicts
        parts = []
        for item in data:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(json.dumps(item))
        return "\n".join(parts).strip()

    # last resort
    try:
        return str(data)
    except Exception:
        return ""


def call_custom_llm(prompt: str, custom_config: Optional[dict]) -> str:
    """
    Call a custom OpenAI-compatible API endpoint using the provided custom configuration.
    """
    if not custom_config:
        raise ValueError(
            "Custom LLM backend selected, but no custom configuration was provided."
        )

    endpoint = custom_config.get("endpoint")
    api_key = custom_config.get("apiKey")
    model_name = custom_config.get("modelName")

    if not endpoint:
        raise ValueError("Custom LLM API Base URL (endpoint) is required.")
    if not model_name:
        raise ValueError("Custom LLM Model Name is required.")

    # Ensure endpoint is correctly formatted.
    # If the endpoint doesn't end with /chat/completions, append it.
    endpoint = endpoint.strip()
    if not endpoint.endswith("/chat/completions"):
        if endpoint.endswith("/"):
            endpoint += "chat/completions"
        else:
            endpoint += "/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a clinical decision support assistant. "
                    "You MUST respond with valid JSON only — no markdown, "
                    "no commentary, no code fences. Just the raw JSON object. "
                    "Always include ALL six fields: condition_id, diagnosis, "
                    "confidence, differential_diagnoses, medicines, reasoning."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }

    start = time.time()
    response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    if not response.ok:
        print(f"[llm_pipeline] Custom LLM error {response.status_code}: {response.text}")
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"]
    print(f"[llm_pipeline] Custom LLM ({model_name}) responded in {elapsed:.1f}s")
    return raw_text


def call_llm(prompt: str, backend: str = LLM_BACKEND, custom_config: Optional[dict] = None) -> str:
    """
    Route to the correct backend based on the selected backend string.

    To switch backend, pass backend = 'groq', 'ollama', 'medgemma', or 'custom'.
    If no backend is passed, falls back to the default LLM_BACKEND.
    """
    normalized = (backend or LLM_BACKEND or "groq").lower()
    if normalized == "groq":
        return call_groq(prompt)
    elif normalized == "ollama":
        return call_ollama(prompt)
    elif normalized == "medgemma":
        return call_medgemma(prompt)
    elif normalized == "custom":
        return call_custom_llm(prompt, custom_config)
    else:
        raise ValueError(
            f"Unknown LLM backend: '{backend}'. "
            "Valid options: 'groq', 'ollama', 'medgemma', 'custom'."
        )


# ---------------------------------------------------------------------
# JSON parsing / extraction
# ---------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_REASONING_RE = re.compile(
    r'"reasoning"\s*:\s*"((?:\\.|[^"\\])*)"',
    re.IGNORECASE | re.DOTALL,
)

# Fields the LLM is expected to always return
REQUIRED_JSON_KEYS = {
    "condition_id", "diagnosis", "confidence",
    "differential_diagnoses", "medicines", "reasoning",
}


def _extract_reasoning(raw_text: str) -> Optional[str]:
    """Extract reasoning field from raw text or generate from diagnosis."""
    match = _REASONING_RE.search(raw_text)
    if not match:
        return None
    reasoning = match.group(1)
    try:
        return bytes(reasoning, "utf-8").decode("unicode_escape").strip()
    except Exception:
        return reasoning.strip()


def _generate_reasoning(parsed: Dict) -> str:
    """
    Auto-generate reasoning from diagnosis/confidence if not provided.
    Fallback for when model omits the field.
    """
    diagnosis_name = parsed.get("diagnosis", {}).get("name", "Unknown condition")
    confidence = parsed.get("confidence", 0.0)
    
    if confidence > 0.7:
        return f"High confidence in {diagnosis_name} based on clinical presentation and knowledge base reference."
    elif confidence > 0.4:
        return f"Moderate confidence in {diagnosis_name}; recommend specialist review for confirmation."
    else:
        return f"Low confidence diagnosis; refer to specialist for further evaluation and confirmation."


def parse_llm_json(raw_text: str) -> Dict:
    """
    Parse the LLM's raw text into a dict.

    Handles common failure modes:
    - Markdown code fences (```json ... ```)
    - Leading/trailing commentary
    - Minor JSON formatting issues
    - Truncated responses (missing trailing fields)

    On total parse failure returns a safe fallback dict that routes
    the case to "refer to specialist" via citation_validator.py.

    On partial parse (valid JSON but missing required keys) logs a
    warning and fills missing keys with safe defaults so the pipeline
    continues gracefully rather than crashing.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    parsed = None

    # Try direct parse first
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back: extract the largest {...} block and try again
    if parsed is None:
        match = _JSON_BLOCK_RE.search(text)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    # Total parse failure — return safe fallback
    if parsed is None:
        print(
            f"[llm_pipeline] WARNING: failed to parse LLM output as JSON. "
            f"Raw output (truncated): {raw_text[:300]!r}"
        )
        return {
            "condition_id": None,
            "diagnosis": {"name": "", "icd10": ""},
            "confidence": 0.0,
            "differential_diagnoses": [],
            "medicines": [],
            "reasoning": (
                "LLM output could not be parsed as valid JSON; "
                "routing to specialist referral as a safety default."
            ),
        }

    # Partial parse — fill missing keys with safe defaults and warn
    missing_keys = REQUIRED_JSON_KEYS - set(parsed.keys())
    if missing_keys:
        # Try to recover reasoning from raw text or generate it
        if "reasoning" in missing_keys:
            reasoning = _extract_reasoning(raw_text)
            if reasoning:
                parsed["reasoning"] = reasoning
                missing_keys.discard("reasoning")
            else:
                # Auto-generate reasoning from diagnosis if available
                parsed["reasoning"] = _generate_reasoning(parsed)
                missing_keys.discard("reasoning")

        if missing_keys:
            print(
                f"[llm_pipeline] WARNING: LLM response missing keys: {missing_keys}. "
                f"Filling with safe defaults. Backend: {LLM_BACKEND}"
            )
            defaults = {
                "condition_id":          None,
                "diagnosis":             {"name": "", "icd10": ""},
                "confidence":            0.0,
                "differential_diagnoses": [],
                "medicines":             [],
                "reasoning":             "(not provided by model)",
            }
            for key in missing_keys:
                parsed[key] = defaults[key]

    return parsed


# ---------------------------------------------------------------------
# End-to-end convenience function
# ---------------------------------------------------------------------

def run_llm_reasoning(
    vitals: dict,
    classifier_probs: List[tuple],
    critical_alert: dict,
    llm_backend: Optional[str] = None,
) -> Dict:
    """
    Full pipeline step:
        1. Build retrieval query from chief complaint + top classifier label
        2. Retrieve RAG context from ChromaDB
        3. Build prompt (vitals + classifier + alert + RAG context)
        4. Call the configured LLM backend
        5. Parse JSON response

    Returns the parsed dict (NOT yet safety-validated).
    Pass result to citation_validator.validate_recommendation().
    """
    top_label  = classifier_probs[0][0] if classifier_probs else ""
    query_text = f"{vitals.get('chief_complaint', '')} {top_label}".strip()

    retrieved_chunks = retrieve_context(query_text)
    prompt = build_prompt(vitals, classifier_probs, critical_alert, retrieved_chunks)

    custom_config = vitals.get("custom_config")
    raw_text = call_llm(prompt, backend=llm_backend, custom_config=custom_config)
    parsed   = parse_llm_json(raw_text)
    return parsed


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick smoke test. Run from project root:
        python llm_pipeline.py

    Requires:
    - chroma_db/ built via build_rag_index.py
    - For groq: GROQ_API_KEY environment variable set
    - For ollama/medgemma: ollama serve running with model pulled
    """
    print(f"Testing backend: {LLM_BACKEND}")

    sample_vitals = {
        "age": 28,
        "sex": "F",
        "bp_systolic": 100,
        "bp_diastolic": 65,
        "blood_sugar": 85,
        "blood_sugar_context": "fasting",
        "weight_kg": 50,
        "height_cm": 155,
        "temperature_c": 36.8,
        "pulse_bpm": 105,
        "chief_complaint": (
            "Fatigue, weakness, pale skin, dizziness on standing, "
            "shortness of breath on exertion, heavy periods"
        ),
    }
    sample_classifier_probs = [
        ("Anaemia (iron-deficiency)", 0.96),
        ("Acute Gastroenteritis", 0.04),
    ]
    sample_alert = {
        "flag": False, "severity": "none",
        "reasons": [], "recommended_action": "",
    }

    result = run_llm_reasoning(sample_vitals, sample_classifier_probs, sample_alert)
    print(json.dumps(result, indent=2))