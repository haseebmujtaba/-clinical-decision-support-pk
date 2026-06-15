"""
run_patient.py
==============
Interactive terminal runner for the CDSS pipeline.
Paste patient JSON, get diagnosis JSON output.

Run: python run_patient.py
"""
import json
import sys
from citation_validator import load_kb
from main import Classifier, run_pipeline

print("=" * 60)
print("CDSS Phase 2 — Terminal Runner")
print("=" * 60)
print("Paste patient JSON below, then press Enter twice to run.")
print("(Copy a 'request' object from test_requests.json)")
print("-" * 60)

lines = []
empty_count = 0

while True:
    try:
        line = input()
        if line == "":
            empty_count += 1
            if empty_count >= 2:
                break
        else:
            empty_count = 0
            lines.append(line)
    except EOFError:
        break

raw = "\n".join(lines).strip()

if not raw:
    print("No input received. Exiting.")
    sys.exit(1)

try:
    patient = json.loads(raw)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
    sys.exit(1)

print("-" * 60)
print("Running pipeline...")
print("-" * 60)

kb = load_kb()
classifier = Classifier()

try:
    result = run_pipeline(patient, classifier, kb)
    print(json.dumps(result, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()