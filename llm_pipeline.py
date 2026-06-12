"""
llm_pipeline.py
===============
Phase 2 - Step 4: LLM reasoning layer.

Calls Mistral 7B via Ollama (100% local/offline) to produce a
structured diagnosis JSON, GROUNDED in:
    - the patient's vitals + chief complaint
    - the XGBoost classifier's probability distribution
    - RAG-retrieved chunks from the WHO EML-based knowledge base
      (via build_rag_index.py / ChromaDB)

The LLM is explicitly instructed to ONLY use conditions and medicines
that appear in the retrieved context, and to output STRICT JSON
matching PROMPT_TEMPLATE's schema. The output is then passed to
citation_validator.py, which is the actual safety gate - this module
does NOT do safety enforcement itself, only retrieval + generation +
JSON parsing.

Requires Ollama running locally with the mistral model pulled:
    ollama pull mistral
    ollama serve   (usually already running as a service)
"""

import json
import os
import re
import time
from typing import Dict, List, Optional

import requests
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "cdss_kb_phase2"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

N_RAG_RESULTS = 8  # retrieved chunks passed to the LLM as context


# ---------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------

_collection = None
_embed_fn = None


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
    Retrieve the top-N most relevant KB chunks for the given query
    (typically the chief complaint + top classifier predictions).

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
markdown, no commentary, no code fences):

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
    Build the full prompt string for Mistral.

    vitals: dict with keys matching VitalsInput fields (age, sex,
            bp_systolic, bp_diastolic, blood_sugar, blood_sugar_context,
            weight_kg, height_cm, temperature_c, pulse_bpm, chief_complaint)
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
# Ollama call
# ---------------------------------------------------------------------

def call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 600) -> str:
    """
    Send the prompt to a locally running Ollama instance and return
    the raw text response.

    On an i5-8500/12GB RAM CPU-only setup, expect ~25-35s per response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # ask Ollama to constrain output to valid JSON
        "options": {
            "temperature": 0.1,  # low temperature: we want consistent,
                                  # grounded clinical output, not creativity
        },
    }
    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    raw_text = data.get("response", "")
    print(f"[llm_pipeline] Ollama ({model}) responded in {elapsed:.1f}s")
    return raw_text


# ---------------------------------------------------------------------
# JSON parsing / extraction
# ---------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_json(raw_text: str) -> Dict:
    """
    Parse the LLM's raw text into a dict. Handles the common failure
    modes of local LLMs: markdown code fences, leading/trailing
    commentary, or minor JSON issues.

    Returns a dict. On total failure, returns a dict with
    "condition_id": None and "confidence": 0.0 so downstream
    citation_validator.py safely routes to "refer to specialist".
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back: extract the largest {...} block and try again
    match = _JSON_BLOCK_RE.search(text)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    print(f"[llm_pipeline] WARNING: failed to parse LLM output as JSON. "
          f"Raw output (truncated): {raw_text[:300]!r}")

    return {
        "condition_id": None,
        "diagnosis": {"name": "", "icd10": ""},
        "confidence": 0.0,
        "differential_diagnoses": [],
        "medicines": [],
        "reasoning": "LLM output could not be parsed as valid JSON; "
                      "routing to specialist referral as a safety default.",
    }


# ---------------------------------------------------------------------
# End-to-end convenience function
# ---------------------------------------------------------------------

def run_llm_reasoning(
    vitals: dict,
    classifier_probs: List[tuple],
    critical_alert: dict,
) -> Dict:
    """
    Full pipeline step: retrieve RAG context -> build prompt -> call
    Mistral via Ollama -> parse JSON. Returns the parsed dict (NOT yet
    safety-validated - pass this to citation_validator.validate_recommendation()).
    """
    # Build a retrieval query from chief complaint + top classifier label
    top_label = classifier_probs[0][0] if classifier_probs else ""
    query_text = f"{vitals.get('chief_complaint', '')} {top_label}".strip()

    retrieved_chunks = retrieve_context(query_text)
    prompt = build_prompt(vitals, classifier_probs, critical_alert, retrieved_chunks)

    raw_text = call_ollama(prompt)
    parsed = parse_llm_json(raw_text)
    return parsed


if __name__ == "__main__":
    # Smoke test (requires chroma_db built via build_rag_index.py
    # and Ollama running with the mistral model pulled)
    sample_vitals = {
        "age": 45,
        "sex": "F",
        "bp_systolic": 110,
        "bp_diastolic": 70,
        "blood_sugar": 95,
        "blood_sugar_context": "random",
        "weight_kg": 60,
        "height_cm": 160,
        "temperature_c": 39.0,
        "pulse_bpm": 96,
        "chief_complaint": "Fever with chills and sweating for 3 days, headache",
    }
    sample_classifier_probs = [
        ("Hypertension", 0.05),
        ("Type 2 Diabetes", 0.03),
        ("No Hypertension", 0.10),
    ]
    sample_alert = {"flag": False, "severity": "none", "reasons": [], "recommended_action": ""}

    result = run_llm_reasoning(sample_vitals, sample_classifier_probs, sample_alert)
    print(json.dumps(result, indent=2))