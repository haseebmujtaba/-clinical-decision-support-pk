"""
debug_pipeline.py
Run this directly to see the full error traceback instead of just "500".
Run: python debug_pipeline.py
"""
import traceback
from citation_validator import load_kb
from main import Classifier, run_pipeline

kb = load_kb()
classifier = Classifier()

sample = {
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
}

try:
    result = run_pipeline(sample, classifier, kb)
    import json
    print(json.dumps(result, indent=2))
except Exception as e:
    print("FULL ERROR:")
    traceback.print_exc()