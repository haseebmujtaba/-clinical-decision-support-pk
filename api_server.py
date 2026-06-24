"""
api_server.py
==============
Phase 2 - Step 7: HTTP API wrapper for the CDSS pipeline.

This exposes main.run_pipeline() as a REST endpoint so the web app
team can call this model from their hospital management system
without needing to know any Python internals.

This file does NOT change the model's logic - it's purely a thin
HTTP wrapper around the existing pipeline (rule_engine -> classifier
-> RAG/LLM -> citation_validator).

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Then POST patient vitals to:
    http://localhost:8000/diagnose

See README_API.md for the full request/response schema and example
curl commands.

IMPORTANT (read before integrating):
- Each /diagnose call triggers a local Mistral 7B inference via
  Ollama. On a CPU-only machine (e.g. i5-8500/12GB RAM, no GPU),
  this takes roughly 1-4 MINUTES per request. This is NOT a bug -
  it is the nature of running a 7B LLM on CPU. The web team should:
    - call this endpoint asynchronously / show a loading state, and
    - NOT expect sub-second responses like a typical REST API.
- Ollama must be running locally (ollama serve) with the "mistral"
  model pulled (ollama pull mistral) BEFORE starting this server.
- The RAG index (chroma_db/) must already be built via
  build_rag_index.py before starting this server.
"""

from typing import List, Optional, Union

import joblib  # noqa: F401  (imported for clearer dependency surfacing)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from citation_validator import load_kb
from main import Classifier, run_pipeline

app = FastAPI(
    title="CDSS Phase 2 - AI Model API",
    description=(
        "Clinical Decision Support model: rule engine + XGBoost "
        "classifier + RAG-grounded LLM + citation/safety validation. "
        "Decision-support only - requires physician sign-off."
    ),
    version="0.2.0",
)

# Allow the web app (running on a different origin, e.g. localhost:3000)
# to call this API directly from the browser during development.
# Restrict allow_origins in production to the actual hospital app domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the classifier + knowledge base ONCE at startup, not per-request.
_classifier = Classifier()
_kb = load_kb()


# ---------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------

class PatientVitals(BaseModel):
    """
    Input schema. Matches rule_engine.VitalsInput fields.
    All fields except chief_complaint are optional so the rule engine
    can still run on partial data, but a real diagnosis needs all of
    them.
    """
    age: Optional[int] = Field(None, example=32, description="Age in years")
    sex: Optional[str] = Field(None, example="F", description="'M' or 'F'")
    bp_systolic: Optional[float] = Field(None, example=110, description="Systolic BP, mmHg")
    bp_diastolic: Optional[float] = Field(None, example=72, description="Diastolic BP, mmHg")
    blood_sugar: Optional[float] = Field(None, example=98, description="Blood sugar, mg/dL")
    blood_sugar_context: Optional[str] = Field(
        "random", example="random", description="'fasting' | 'random' | 'post_meal'"
    )
    weight_kg: Optional[float] = Field(None, example=58, description="Weight in kg")
    height_cm: Optional[float] = Field(None, example=162, description="Height in cm")
    temperature_c: Optional[float] = Field(None, example=39.2, description="Body temperature, Celsius")
    pulse_bpm: Optional[float] = Field(None, example=102, description="Pulse, beats per minute")
    llm_backend: Optional[str] = Field(
        "groq",
        example="groq",
        description="LLM backend to use: 'groq', 'ollama', or 'medgemma'"
    )
    chief_complaint: str = Field(
        "", example="Fever with chills and sweating for 3 days, severe headache",
        description="Free-text description of the patient's main complaint"
    )

    class Config:
        json_schema_extra = {
            "example": {
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
                "chief_complaint": "Fever with chills and sweating for 3 days, severe headache, body aches",
            }
        }


class CriticalAlert(BaseModel):
    flag: bool
    severity: str
    reasons: List[str]
    recommended_action: str


class DiagnosisInfo(BaseModel):
    name: Optional[str] = None
    icd10: Optional[str] = None


class Medicine(BaseModel):
    name: str
    dosage_form: str
    dose_instruction: str
    tier: Union[int, List[int]]
    citation: str
    drap_check: dict
    flag: Optional[str] = None


class ClassifierResult(BaseModel):
    label: str
    probability: float


class DiagnosisResponse(BaseModel):
    critical_alert: CriticalAlert
    diagnosis: dict  # {} when refer_to_specialist, else DiagnosisInfo-shaped
    confidence: float
    differential_diagnoses: List[dict]
    medicines: List[Medicine]
    treatment_plan_notes: Optional[str] = None
    confidence_notes: Optional[str] = None
    action: str
    refer_to_specialist: bool
    refer_reason: Optional[str] = None
    classifier_output: List[ClassifierResult]
    llm_reasoning: str


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.get("/")
def root():
    """Basic health/info endpoint."""
    return {
        "service": "CDSS Phase 2 AI Model API",
        "status": "running",
        "note": (
            "POST patient vitals to /diagnose. Each request takes "
            "roughly 1-4 minutes on CPU-only hardware due to local "
            "LLM inference (Mistral 7B via Ollama)."
        ),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """
    Health check. Confirms the classifier and knowledge base loaded
    successfully. Does NOT check Ollama connectivity (that is only
    verified on an actual /diagnose call, since pinging it would
    itself trigger model loading).
    """
    return {
        "status": "ok",
        "classifier_loaded": _classifier.model is not None,
        "kb_conditions_loaded": len(_kb.get("conditions", [])),
    }


@app.post("/diagnose", response_model=DiagnosisResponse)
def diagnose(patient: PatientVitals):
    """
    Run the full CDSS pipeline (rule engine -> classifier -> RAG/LLM
    -> citation validator) for a single patient and return the final
    decision-support JSON.

    NOTE: This is a SYNCHRONOUS, BLOCKING call that takes ~1-4 minutes
    on CPU-only hardware. The calling web app should show a loading
    indicator and use a generous request timeout (recommend >= 300s).
    """
    try:
        result = run_pipeline(
            patient.model_dump(),
            _classifier,
            _kb,
            llm_backend=patient.llm_backend,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {type(e).__name__}: {e}",
        )
    return result


@app.get("/conditions")
def list_conditions():
    """
    Returns the list of conditions currently covered by the knowledge
    base (condition_id, name, ICD-10). Useful for the web team to
    understand current model coverage / scope.
    """
    return [
        {
            "condition_id": c["condition_id"],
            "condition_name": c["condition_name"],
            "icd10": c.get("icd10", ""),
        }
        for c in _kb.get("conditions", [])
    ]
