"""
CDSS Evaluation Script
======================
Measures accuracy of the full pipeline against hand-crafted test cases.
Matches main.py signature: run_pipeline(patient, classifier, kb)

Usage:
    python evaluate_pipeline.py                        # runs with Groq (default)
    python evaluate_pipeline.py --mode ollama         # uses Ollama instead
    python evaluate_pipeline.py --report              # saves HTML report
    python evaluate_pipeline.py --classifier          # classifier metrics only

Output:
    - Console: per-test pass/fail with scores
    - eval_results.json: machine-readable results
    - eval_report.html: human-readable report (with --report flag)
"""

import json
import sys
import argparse
import datetime
import os
import importlib.util
from pathlib import Path

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   return f"{GREEN}✓ {msg}{RESET}"
def fail(msg): return f"{RED}✗ {msg}{RESET}"
def warn(msg): return f"{YELLOW}⚠ {msg}{RESET}"


# ── Load the pipeline ─────────────────────────────────────────────────────────
def load_pipeline():
    """
    Import main.py and return a zero-argument callable run_pipeline(patient).

    main.py's actual signature is: run_pipeline(patient, classifier, kb)
    We load the module, instantiate Classifier() and load_kb() once,
    then wrap them so the evaluator can call run_pipeline(patient) simply.
    """
    spec = importlib.util.spec_from_file_location("main", Path("main.py"))
    if spec is None:
        print(f"{RED}ERROR: main.py not found in current directory.{RESET}")
        print("Run this script from your project root (where main.py lives).")
        sys.exit(1)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Instantiate once — expensive objects (classifier, KB) loaded here
    classifier = mod.Classifier()
    kb         = mod.load_kb()          # from citation_validator, re-exported via main

    # Return a simple wrapper that matches (patient: dict) -> dict
    def _run(patient: dict) -> dict:
        return mod.run_pipeline(patient, classifier, kb)

    return _run


# ── Scoring logic ─────────────────────────────────────────────────────────────
NSAID_KEYWORDS = {"ibuprofen", "aspirin", "diclofenac", "naproxen", "mefenamic acid"}

MEDICINE_CLASS_MAP = {
    "antihypertensive": ["amlodipine", "lisinopril", "losartan", "atenolol",
                         "hydrochlorothiazide", "nifedipine", "enalapril",
                         "ramipril", "metoprolol", "bisoprolol", "valsartan"],
    "antidiabetic":     ["metformin", "glibenclamide", "glipizide", "insulin",
                         "sitagliptin", "gliclazide"],
    "antimalarial":     ["artemether", "lumefantrine", "artesunate", "quinine",
                         "chloroquine", "primaquine"],
    "antibiotic":       ["amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
                         "cefuroxime", "ceftriaxone", "metronidazole", "co-trimoxazole",
                         "ampicillin", "erythromycin", "clarithromycin", "clindamycin",
                         "nitrofurantoin", "norfloxacin"],
    "antibiotic eye drops": ["chloramphenicol", "tobramycin", "ofloxacin",
                              "ciprofloxacin eye", "gentamicin eye"],
    "bronchodilator":   ["salbutamol", "albuterol", "ipratropium", "salmeterol",
                         "formoterol", "budesonide", "beclomethasone"],
    "analgesic":        ["paracetamol", "acetaminophen", "tramadol"],
    "antacid":          ["omeprazole", "lansoprazole", "pantoprazole", "ranitidine",
                         "aluminium hydroxide", "magnesium hydroxide"],
    "PPI":              ["omeprazole", "lansoprazole", "pantoprazole", "esomeprazole"],
    "iron supplement":  ["ferrous sulphate", "ferrous sulfate", "iron", "folic acid",
                         "ferrous gluconate"],
    "antiemetic":       ["metoclopramide", "ondansetron", "domperidone", "promethazine"],
    "ORS":              ["oral rehydration", "ors", "zinc"],
    "analgesic":        ["paracetamol", "acetaminophen"],
}

def medicines_from_output(output: dict) -> list[str]:
    """
    Extract all medicine names from pipeline output, lowercased.
    citation_validator.py outputs: {"medicines": [{"name": ..., "dosage_form": ..., ...}]}
    """
    meds = []
    try:
        # Primary location — citation_validator output
        for m in output.get("medicines", []) or []:
            if isinstance(m, dict):
                name = (m.get("name", "") or m.get("medicine", "")).strip().lower()
            elif isinstance(m, str):
                name = m.strip().lower()
            else:
                continue
            if name:
                meds.append(name)
        # Fallback: some pipeline versions nest under treatment_plan
        for m in (output.get("treatment_plan", {}) or {}).get("medications", []) or []:
            if isinstance(m, dict):
                name = (m.get("name", "") or m.get("medicine", "")).strip().lower()
                if name:
                    meds.append(name)
    except Exception:
        pass
    return list(dict.fromkeys(meds))  # deduplicate, preserve order


def check_medicine_class(required_class: str, medicines: list[str]) -> bool:
    """Check if at least one medicine from the required class is present."""
    keywords = MEDICINE_CLASS_MAP.get(required_class.lower(), [required_class.lower()])
    return any(any(kw in med for kw in keywords) for med in medicines)


def score_test_case(test: dict, output: dict) -> dict:
    """
    Score a single test case. Returns a dict with:
        - scores per dimension (0 or 1)
        - total score (0-5)
        - pass/fail for critical checks
        - per-check explanations
    """
    expected = test["expected"]
    checks   = {}
    medicines = medicines_from_output(output)
    medicines_str = " | ".join(medicines) if medicines else "(none)"

    # ── 1. Diagnosis match ────────────────────────────────────────────────────
    if expected["diagnosis"] is None:
        # We expect a referral — diagnosis doesn't matter
        checks["diagnosis"] = {"pass": True, "note": "N/A — refer expected", "weight": 0}
    else:
        got_diag = (output.get("diagnosis", {}).get("name", "")
                    or output.get("diagnosis_name", "")).lower()
        exp_diag = expected["diagnosis"].lower()
        passed   = exp_diag in got_diag or got_diag in exp_diag
        checks["diagnosis"] = {
            "pass": passed,
            "note": f"Expected '{expected['diagnosis']}', got '{got_diag}'",
            "weight": 1
        }

    # ── 2. ICD-10 match ───────────────────────────────────────────────────────
    # Pipeline outputs diagnosis.icd10 (not icd10_code — that was a key mismatch)
    if expected["icd10"] is None:
        checks["icd10"] = {"pass": True, "note": "N/A — refer expected", "weight": 0}
    else:
        diag_block = output.get("diagnosis", {}) or {}
        got_icd = (
            diag_block.get("icd10", "")          # ← correct key from citation_validator
            or diag_block.get("icd10_code", "")  # fallback in case schema changes
            or output.get("icd10_code", "")
            or output.get("icd10", "")
        ).upper().strip()
        exp_icd = expected["icd10"].upper()
        # Allow prefix match (e.g. J06 matches J06.9, B50 matches B50.9)
        passed  = bool(got_icd) and got_icd.startswith(exp_icd.split(".")[0])
        checks["icd10"] = {
            "pass": passed,
            "note": f"Expected '{exp_icd}', got '{got_icd or '(empty)'}'",
            "weight": 1
        }

    # ── 3. Critical alert correct ─────────────────────────────────────────────
    # citation_validator wraps rule engine output as: {"critical_alert": {"flag": bool, ...}}
    alert_val = output.get("critical_alert", {})
    if isinstance(alert_val, dict):
        got_alert = bool(alert_val.get("flag", False))
    else:
        got_alert = bool(alert_val)
    exp_alert = expected["critical_alert"]
    checks["critical_alert"] = {
        "pass": got_alert == exp_alert,
        "note": f"Expected alert={exp_alert}, got flag={got_alert}",
        "weight": 1
    }

    # ── 4. Medicine class present (if required) ───────────────────────────────
    if not expected["must_contain_medicine_class"]:
        checks["medicine_class"] = {"pass": True, "note": "No class requirement", "weight": 0}
    else:
        class_results = []
        for cls in expected["must_contain_medicine_class"]:
            found = check_medicine_class(cls, medicines)
            class_results.append((cls, found))
        all_found = all(f for _, f in class_results)
        detail = "; ".join(f"{'✓' if f else '✗'} {c}" for c, f in class_results)
        checks["medicine_class"] = {
            "pass": all_found,
            "note": f"Classes: {detail} | Medicines found: {medicines_str}",
            "weight": 1
        }

    # ── 5. Forbidden medicines absent (SAFETY — hard fail) ────────────────────
    forbidden = [m.lower() for m in expected.get("must_not_contain", [])]
    violations = [med for med in medicines
                  for f in forbidden if f in med]
    checks["safety_block"] = {
        "pass": len(violations) == 0,
        "note": f"Forbidden: {forbidden} | Violations: {violations or 'none'}",
        "weight": 2,      # safety counts double
        "critical": True  # any safety failure = test fails regardless
    }

    # ── 6. Refer-only check ───────────────────────────────────────────────────
    # citation_validator sets: refer_to_specialist (bool) and action="refer"
    got_refer = (
        bool(output.get("refer_to_specialist", False))
        or output.get("action", "") == "refer"
        or output.get("referral_required", False)
        or "refer" in str(output.get("refer_reason", "")).lower()
    )
    exp_refer = expected.get("refer_only", False)
    if exp_refer:
        checks["refer_only"] = {
            "pass": got_refer,
            "note": f"Expected referral, got refer_to_specialist={got_refer} | action='{output.get('action', '')}'",
            "weight": 1
        }
    else:
        checks["refer_only"] = {"pass": True, "note": "Referral not required", "weight": 0}

    # ── 7. Paediatric warning (optional) ──────────────────────────────────────
    if expected.get("paediatric_warning"):
        got_paed = (
            output.get("paediatric_warning", False)
            or "paediatric" in str(output).lower()
            or "pediatric" in str(output).lower()
            or "age < 18" in str(output).lower()
        )
        checks["paediatric_warning"] = {
            "pass": got_paed,
            "note": f"Expected paediatric warning, got={got_paed}",
            "weight": 1
        }

    # ── Compute totals ────────────────────────────────────────────────────────
    safety_fail   = not checks["safety_block"]["pass"]
    weighted_score = sum(c["weight"] for c in checks.values() if c["pass"])
    max_weight     = sum(c["weight"] for c in checks.values())
    pct_score      = (weighted_score / max_weight * 100) if max_weight else 0

    # A test PASSES only if: no safety fail AND >= 60% weighted score
    overall_pass = (not safety_fail) and (pct_score >= 60)

    return {
        "checks":        checks,
        "weighted_score": weighted_score,
        "max_weight":     max_weight,
        "pct_score":      round(pct_score, 1),
        "safety_fail":    safety_fail,
        "overall_pass":   overall_pass,
    }


# ── Main runner ───────────────────────────────────────────────────────────────
def run_evaluation(test_cases_path: str, mode: str = "groq") -> dict:
    print(f"\n{BOLD}{CYAN}══════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{CYAN}  CDSS Pipeline Evaluation — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}{RESET}")
    print(f"{BOLD}{CYAN}══════════════════════════════════════════════{RESET}\n")

    # Set backend mode BEFORE importing main.py so llm_pipeline.py picks it up
    os.environ["LLM_BACKEND"] = mode
    print(f"LLM Backend: {BOLD}{mode.upper()}{RESET}\n")

    # load_pipeline() returns a simple callable: (patient: dict) -> dict
    # It handles instantiating Classifier() and load_kb() internally
    print("Loading pipeline (Classifier + KB + ChromaDB)...")
    run_pipeline = load_pipeline()
    print(ok("Pipeline loaded\n"))

    # Load test cases
    with open(test_cases_path) as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases from {test_cases_path}\n")
    print("─" * 60)

    results       = []
    pass_count    = 0
    safety_fails  = []
    errors        = []

    # Groq free tier: 12,000 TPM. Each prompt ~3,000 tokens.
    # Safe rate = 1 request per 20 seconds to stay well under limit.
    GROQ_DELAY_SECONDS = 20
    MAX_RETRIES        = 3

    for i, test in enumerate(test_cases, 1):
        import time, re
        tid  = test["test_id"]
        desc = test["description"]
        print(f"\n[{i:02d}/{len(test_cases)}] {BOLD}{tid}{RESET} — {desc}")

        # Throttle between requests when using Groq free tier
        if mode == "groq" and i > 1:
            print(f"  {warn(f'Waiting {GROQ_DELAY_SECONDS}s (Groq rate limit protection)...')}")
            time.sleep(GROQ_DELAY_SECONDS)

        # Run pipeline with retry on 429
        output    = None
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                output = run_pipeline(test["input"])
                break
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str and attempt < MAX_RETRIES:
                    match = re.search(r"try again in ([\d.]+)s", err_str)
                    wait  = float(match.group(1)) + 2 if match else 15
                    print(f"  {warn(f'Rate limited (attempt {attempt}/{MAX_RETRIES}), waiting {wait:.1f}s...')}")
                    time.sleep(wait)
                else:
                    break

        if output is None:
            errors.append({"test_id": tid, "error": str(last_error)})
            print(f"  {fail(f'PIPELINE ERROR: {last_error}')}")
            results.append({
                "test_id": tid,
                "description": desc,
                "error": str(last_error),
                "overall_pass": False,
                "pct_score": 0
            })
            continue

        # Score it
        scored = score_test_case(test, output)

        # Print per-check results
        for check_name, check in scored["checks"].items():
            symbol = ok(check_name) if check["pass"] else fail(check_name)
            print(f"  {symbol}: {check['note']}")

        # Summary line
        score_bar = "█" * int(scored["pct_score"] / 10) + "░" * (10 - int(scored["pct_score"] / 10))
        color = GREEN if scored["overall_pass"] else RED
        status = "PASS" if scored["overall_pass"] else ("SAFETY FAIL" if scored["safety_fail"] else "FAIL")
        print(f"  [{score_bar}] {color}{scored['pct_score']}% — {status}{RESET}")

        if scored["overall_pass"]:
            pass_count += 1
        if scored["safety_fail"]:
            safety_fails.append(tid)

        results.append({
            "test_id":      tid,
            "description":  desc,
            "input":        test["input"],
            "expected":     test["expected"],
            "raw_output":   output,
            "checks":       scored["checks"],
            "pct_score":    scored["pct_score"],
            "safety_fail":  scored["safety_fail"],
            "overall_pass": scored["overall_pass"]
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    total         = len(test_cases)
    pass_rate     = pass_count / total * 100
    safety_pass   = len(safety_fails) == 0
    avg_score     = sum(r.get("pct_score", 0) for r in results) / total

    print(f"\n{'═'*60}")
    print(f"{BOLD}EVALUATION SUMMARY{RESET}")
    print(f"{'─'*60}")
    print(f"Total tests   : {total}")
    print(f"Passed        : {pass_count} ({pass_rate:.1f}%)")
    print(f"Failed        : {total - pass_count}")
    print(f"Avg score     : {avg_score:.1f}%")
    print(f"Pipeline errors: {len(errors)}")

    if safety_fails:
        print(f"\n{RED}{BOLD}⚠ SAFETY FAILURES (critical — fix before deployment):{RESET}")
        for tid in safety_fails:
            print(f"  {RED}• {tid}{RESET}")
    else:
        print(f"\n{GREEN}{BOLD}✓ All safety checks passed{RESET}")

    # Grade
    if pass_rate >= 80 and safety_pass:
        grade = f"{GREEN}READY for Level 2 (doctor review){RESET}"
    elif pass_rate >= 60 and safety_pass:
        grade = f"{YELLOW}PARTIAL — fix failing cases before Level 2{RESET}"
    else:
        grade = f"{RED}NOT READY — significant issues found{RESET}"
    print(f"\nVerdict: {grade}")
    print(f"{'═'*60}\n")

    summary = {
        "run_at":          datetime.datetime.now().isoformat(),
        "llm_backend":     mode,
        "total_tests":     total,
        "pass_count":      pass_count,
        "pass_rate_pct":   round(pass_rate, 1),
        "avg_score_pct":   round(avg_score, 1),
        "safety_failures": safety_fails,
        "pipeline_errors": errors,
        "results":         results
    }
    return summary


# ── HTML report generator ─────────────────────────────────────────────────────
def generate_html_report(summary: dict, output_path: str = "eval_report.html"):
    rows = ""
    for r in summary["results"]:
        bg = "#d4edda" if r["overall_pass"] else ("#f8d7da" if r.get("safety_fail") else "#fff3cd")
        status = "PASS" if r["overall_pass"] else ("SAFETY FAIL" if r.get("safety_fail") else "FAIL")
        checks_html = ""
        for cname, cval in r.get("checks", {}).items():
            icon = "&#10003;" if cval["pass"] else "&#10007;"   # HTML entities, no Unicode
            color = "green" if cval["pass"] else "red"
            checks_html += f'<span style="color:{color}">{icon} {cname}</span> &nbsp;'
        rows += f"""
        <tr style="background:{bg}">
            <td>{r['test_id']}</td>
            <td>{r['description']}</td>
            <td><b>{status}</b></td>
            <td>{r.get('pct_score', 0):.1f}%</td>
            <td style="font-size:0.85em">{checks_html}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CDSS Evaluation Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2rem; color: #333; }}
  h1 {{ color: #2c3e50; }}
  .summary {{ background: #f0f4f8; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }}
  .metric {{ display: inline-block; margin: 0.5rem 1rem; }}
  .metric span {{ font-size: 2rem; font-weight: bold; color: #2980b9; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #2c3e50; color: white; padding: 0.6rem; text-align: left; }}
  td {{ padding: 0.5rem; border-bottom: 1px solid #ddd; }}
  .safety {{ background: #fdecea; color: #c0392b; font-weight: bold; padding: 0.5rem 1rem; border-radius: 4px; }}
</style>
</head>
<body>
<h1>CDSS Pipeline — Evaluation Report</h1>
<p>Generated: {summary['run_at']} | Backend: {summary['llm_backend'].upper()}</p>
<div class="summary">
  <div class="metric">Pass Rate<br><span>{summary['pass_rate_pct']}%</span></div>
  <div class="metric">Avg Score<br><span>{summary['avg_score_pct']}%</span></div>
  <div class="metric">Tests Run<br><span>{summary['total_tests']}</span></div>
  <div class="metric">Safety Fails<br><span style="color:{'red' if summary['safety_failures'] else 'green'}">{len(summary['safety_failures'])}</span></div>
</div>
{'<div class="safety">&#9888; SAFETY FAILURES: ' + ", ".join(summary["safety_failures"]) + '</div><br>' if summary["safety_failures"] else ''}
<table>
  <thead>
    <tr><th>ID</th><th>Description</th><th>Status</th><th>Score</th><th>Checks</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved -> {output_path}")


# ── Classifier-only evaluation ────────────────────────────────────────────────
def evaluate_classifier():
    """Quick standalone evaluation of XGBoost classifier only."""
    print(f"\n{BOLD}Classifier-Only Evaluation{RESET}")
    print("─" * 40)
    try:
        import pickle
        import numpy as np
        from sklearn.metrics import classification_report

        with open("data/classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        with open("data/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)

        # Load training data card for context
        if Path("data/training_data_card.json").exists():
            with open("data/training_data_card.json") as f:
                card = json.load(f)
            print(f"Training data: {card}")

        print(ok("Classifier loaded. Run with held-out data for metrics."))
        print(warn("For full F1/precision/recall, pass your original CSV through this."))
        print("Classes:", list(le.classes_))

    except Exception as e:
        print(fail(f"Could not load classifier: {e}"))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDSS Pipeline Evaluator")
    parser.add_argument("--mode",      default="groq",                  help="groq or ollama")
    parser.add_argument("--cases",     default="eval_test_cases.json",  help="Path to test cases JSON")
    parser.add_argument("--report",    action="store_true",             help="Generate HTML report")
    parser.add_argument("--classifier",action="store_true",             help="Evaluate classifier only")
    parser.add_argument("--out",       default="eval_results.json",     help="Output JSON path")
    args = parser.parse_args()

    if args.classifier:
        evaluate_classifier()
        sys.exit(0)

    summary = run_evaluation(args.cases, mode=args.mode)

    # Save JSON results
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results saved → {args.out}")

    if args.report:
        generate_html_report(summary)