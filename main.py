"""
main.py
=======
Phase 2 - Step 6: End-to-end pipeline wiring.

Pipeline order (per project spec):
    1. Rule engine (rule_engine.py)        - runs FIRST, on raw vitals
    2. ML classifier (XGBoost, classifier.pkl + label_encoder.pkl)
    3. RAG retrieval + LLM reasoning (llm_pipeline.py, Mistral via Ollama)
    4. Citation validation + drug safety check (citation_validator.py)
    5. Final JSON output

This file is the single entry point a separate web-app team can call
(e.g. via a thin FastAPI/Flask wrapper - not built in Phase 2) to go
from raw patient vitals to the final decision-support JSON.

Usage:
    python main.py                 # runs built-in sample patients
    python main.py patient.json    # runs a single patient from a JSON file
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from rule_engine import VitalsInput, run_rule_engine
from llm_pipeline import run_llm_reasoning
from citation_validator import validate_recommendation, load_kb

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLASSIFIER_PATH = os.path.join(DATA_DIR, "classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")

# Feature order MUST match the training script's merged schema:
# bp_systolic, bp_diastolic, blood_sugar, weight_kg, height_cm, bmi,
# temperature_c, pulse_bpm, age, sex
CLASSIFIER_FEATURES = [
    "bp_systolic", "bp_diastolic", "blood_sugar", "weight_kg",
    "height_cm", "bmi", "temperature_c", "pulse_bpm", "age", "sex",
]


# ---------------------------------------------------------------------
# Step 2: ML classifier
# ---------------------------------------------------------------------

class Classifier:
    """
    Thin wrapper around the saved XGBoost classifier + label encoder.

    If the .pkl files aren't present (e.g. running this module in an
    environment without the Day3-5 training artifacts), falls back
    to returning an empty probability list so the rest of the
    pipeline still runs - the LLM prompt explicitly handles "no
    classifier output available".
    """

    def __init__(self, classifier_path: str = CLASSIFIER_PATH,
                 label_encoder_path: str = LABEL_ENCODER_PATH):
        self.model = None
        self.label_encoder = None
        if os.path.exists(classifier_path) and os.path.exists(label_encoder_path):
            try:
                self.model = joblib.load(classifier_path)
                self.label_encoder = joblib.load(label_encoder_path)
            except Exception as e:
                print(f"[main] WARNING: failed to load classifier artifacts: {e}")
        else:
            print(f"[main] WARNING: classifier.pkl / label_encoder.pkl not found "
                  f"in {DATA_DIR} - proceeding without classifier signal.")

    def _bmi(self, weight_kg, height_cm):
        try:
            if weight_kg and height_cm:
                h_m = float(height_cm) / 100.0
                if h_m > 0:
                    return round(float(weight_kg) / (h_m * h_m), 1)
        except (TypeError, ValueError):
            pass
        return 0.0

    def predict_proba(self, vitals: dict) -> List[Tuple[str, float]]:
        """
        Returns a list of (label, probability) tuples sorted by
        probability descending. Empty list if classifier unavailable.
        """
        if self.model is None or self.label_encoder is None:
            return []

        bmi = vitals.get("bmi") or self._bmi(vitals.get("weight_kg"), vitals.get("height_cm"))

        # sex encoding: training schema assumed numeric encoding for
        # sex (e.g. M=1, F=0). Adjust here if the Day3-5 script used a
        # different convention - this is the one integration point
        # most likely to need tweaking once classifier.pkl internals
        # are confirmed.
        sex_raw = str(vitals.get("sex", "")).strip().upper()
        sex_val = 1 if sex_raw == "M" else 0

        row = {
            "bp_systolic": vitals.get("bp_systolic", 0) or 0,
            "bp_diastolic": vitals.get("bp_diastolic", 0) or 0,
            "blood_sugar": vitals.get("blood_sugar", 0) or 0,
            "weight_kg": vitals.get("weight_kg", 0) or 0,
            "height_cm": vitals.get("height_cm", 0) or 0,
            "bmi": bmi or 0,
            "temperature_c": vitals.get("temperature_c", 0) or 0,
            "pulse_bpm": vitals.get("pulse_bpm", 0) or 0,
            "age": vitals.get("age", 0) or 0,
            "sex": sex_val,
        }

        X = pd.DataFrame([row], columns=CLASSIFIER_FEATURES)

        try:
            probs = self.model.predict_proba(X)[0]
        except Exception as e:
            print(f"[main] WARNING: classifier.predict_proba failed: {e}")
            return []

        labels = self.label_encoder.inverse_transform(np.arange(len(probs)))
        pairs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
        return [(str(label), float(prob)) for label, prob in pairs]


# ---------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------

def run_pipeline(patient: dict, classifier: Classifier, kb: dict) -> dict:
    """
    Run the full CDSS pipeline for a single patient record.

    `patient` is a dict matching VitalsInput's fields:
        bp_systolic, bp_diastolic, blood_sugar, blood_sugar_context,
        weight_kg, height_cm, temperature_c, pulse_bpm, age, sex,
        chief_complaint

    Returns the final JSON-ready dict.
    """
    # --- Step 1: Rule engine (runs first, on raw vitals) ---
    vitals_obj = VitalsInput(
        bp_systolic=patient.get("bp_systolic"),
        bp_diastolic=patient.get("bp_diastolic"),
        blood_sugar=patient.get("blood_sugar"),
        blood_sugar_context=patient.get("blood_sugar_context", "random"),
        weight_kg=patient.get("weight_kg"),
        height_cm=patient.get("height_cm"),
        temperature_c=patient.get("temperature_c"),
        pulse_bpm=patient.get("pulse_bpm"),
        age=patient.get("age"),
        sex=patient.get("sex"),
        chief_complaint=patient.get("chief_complaint", ""),
    )
    critical_alert = run_rule_engine(vitals_obj).to_dict()

    # --- Step 2: ML classifier (structured vitals only) ---
    classifier_probs = classifier.predict_proba(patient)

    # --- Step 3: RAG retrieval + LLM reasoning (Mistral via Ollama) ---
    llm_output = run_llm_reasoning(patient, classifier_probs, critical_alert)

    # --- Step 4 + 5: Citation validation, drug safety, confidence
    #     gating -> final JSON ---
    final = validate_recommendation(
        llm_output=llm_output,
        rule_engine_alert=critical_alert,
        chief_complaint=patient.get("chief_complaint", ""),
        kb=kb,
    )

    # Attach classifier output for transparency / audit trail.
    final["classifier_output"] = [
        {"label": label, "probability": round(prob, 4)}
        for label, prob in classifier_probs[:5]
    ]
    final["llm_reasoning"] = llm_output.get("reasoning", "")

    return final


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

SAMPLE_PATIENTS = [
    {
        "_label": "Sample 1: likely malaria, dengue in differential (NSAID block test)",
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
    },
    {
        "_label": "Sample 2: hypertensive crisis (critical alert test)",
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
        "chief_complaint": "Severe headache and blurred vision",
    },
    {
        "_label": "Sample 3: T2DM-suspicious vitals",
        "age": 50,
        "sex": "M",
        "bp_systolic": 128,
        "bp_diastolic": 82,
        "blood_sugar": 215,
        "blood_sugar_context": "random",
        "weight_kg": 90,
        "height_cm": 170,
        "temperature_c": 37.0,
        "pulse_bpm": 84,
        "chief_complaint": "Increased thirst, frequent urination, fatigue for 2 weeks",
    },
]


def main():
    kb = load_kb()
    classifier = Classifier()

    if len(sys.argv) > 1:
        # Single patient from a JSON file
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            patient = json.load(f)
        result = run_pipeline(patient, classifier, kb)
        print(json.dumps(result, indent=2))
        return

    # Built-in sample patients
    for sample in SAMPLE_PATIENTS:
        label = sample.pop("_label")
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        result = run_pipeline(sample, classifier, kb)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
