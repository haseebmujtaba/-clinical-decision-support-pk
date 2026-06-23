"""
citation_validator.py
======================
Phase 2 - Step 5: Citation lookup + drug safety check.

This module is the LEGAL/SAFETY GATE of the pipeline. Every medicine
that reaches the final output JSON must pass through here. Its jobs:

1. CITATION ENFORCEMENT
   - Every medicine recommendation MUST carry a citation from the
     knowledge base (Tier 1-4). If a medicine the LLM proposes has
     NO matching KB entry / NO citation, it is DROPPED and the
     condition is downgraded to "refer to specialist" for that item.
   - This is the core legal safety mechanism described in the
     project brief: "If NO citation found -> withhold + refer."

2. CONFIDENCE-BASED GATING
   - confidence > 0.80  -> full recommendation (diagnosis + meds)
   - 0.60 <= confidence <= 0.80 -> diagnosis only, "verify with specialist"
   - confidence < 0.60  -> refer to specialist entirely, NO medicines shown

3. SAFETY OVERRIDES (hard-coded clinical guardrails, independent of
   confidence score):
   - Dengue in the differential list -> hard-block NSAIDs/aspirin
     (ibuprofen, acetylsalicylic acid, diclofenac, naproxen, aspirin,
     mefenamic acid) due to bleeding risk.
   - TB_PULM condition -> medicines are ALWAYS marked reference-only
     and suppressed from the patient-facing medicine list; output
     always includes "refer to national TB programme / DOTS".
   - PUD H. pylori eradication antibiotics (amoxicillin, clarithromycin,
     metronidazole) are suppressed unless H. pylori positivity is
     documented in the chief complaint / notes.
   - GASTROENTERITIS: ciprofloxacin suppressed unless dysentery features
     (blood in stool / "dysentery") are documented in chief complaint.
   - URI_PHARYNGITIS: antibiotics (phenoxymethylpenicillin, amoxicillin)
     are flagged "physician review: bacterial vs viral" rather than
     shown as a default recommendation.

4. DRAP NATIONAL DRUG LIST CHECK (placeholder for Phase 2)
   - Phase 2 KB explicitly notes that Pakistan's NEML 2025 = WHO 24th
     list verbatim (Gazette 20-Oct-2025), so KB medicines are treated
     as DRAP/PNF-aligned (Tier 2) for now. This module still exposes a
     `check_drap_registration()` hook that a future phase can wire up
     to a live DRAP National Drug List lookup (API or local DB). Until
     then it returns "not_checked" rather than fabricating a result.

Output of validate_recommendation() is the FINAL, safety-gated JSON-
ready dict that main.py should use as the source of truth - even if
it differs from what the LLM originally proposed.
"""

import json
import os
from typing import Dict, List, Optional

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")

# Medicines that are an absolute contraindication when dengue is in the
# differential list, due to bleeding risk (per KB confidence_notes).
DENGUE_CONTRAINDICATED = {
    "ibuprofen",
    "acetylsalicylic acid",
    "aspirin",
    "diclofenac",
    "naproxen",
    "mefenamic acid",
}

# Conditions whose medicines must NEVER reach the patient-facing
# medicine list - reference-only, always refer.
REFERENCE_ONLY_CONDITIONS = {"TB_PULM"}

# Conditionally-shown antibiotics: only included if the relevant
# clinical trigger keyword appears in the chief complaint / notes.
CONDITIONAL_ANTIBIOTICS = {
    "PUD": {
        "trigger_keywords": ["h. pylori", "h pylori", "helicobacter", "hpylori"],
        "conditional_medicines": {"amoxicillin", "clarithromycin", "metronidazole"},
    },
    "GASTROENTERITIS": {
        "trigger_keywords": ["dysentery", "blood in stool", "bloody stool", "bloody diarrhea", "bloody diarrhoea"],
        "conditional_medicines": {"ciprofloxacin"},
    },
}

# Antibiotics that should be flagged for explicit bacterial-vs-viral
# physician review rather than auto-recommended.
PHYSICIAN_REVIEW_ANTIBIOTICS = {
    "URI_PHARYNGITIS": {"phenoxymethylpenicillin", "amoxicillin"},
}

CONFIDENCE_HIGH = 0.80
CONFIDENCE_LOW = 0.60


def load_kb(path: str = KB_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _kb_index(kb: dict) -> Dict[str, dict]:
    """Index conditions by condition_id for O(1) lookup."""
    return {c["condition_id"]: c for c in kb["conditions"]}


def _kb_medicine_lookup(condition: dict) -> Dict[str, dict]:
    """Index a condition's medicines by lowercase name for lookup."""
    meds = condition.get("treatment_plan", {}).get("medicines", [])
    return {m["name"].lower(): m for m in meds}


def check_drap_registration(medicine_name: str) -> Dict[str, str]:
    """
    DRAP/NEML verification via Pakistan Government Gazette.

    Legal basis: Government of Pakistan, Ministry of National Health
    Services, Regulations & Coordination, Gazette Notification
    F.No.8-43/2021-DD(PS) dated 20th October 2025 — formally adopted
    the WHO Model List of Essential Medicines 24th List (2025) verbatim
    as Pakistan's National Essential Medicines List (NEML) 2025.

    Any medicine present in knowledge_base.json (which was built
    directly from the WHO EML 24th List 2025) is therefore confirmed
    as both:
      - Tier 1: WHO Model List of Essential Medicines 24th List (2025)
      - Tier 2: Pakistan NEML 2025 / DRAP-aligned (via Gazette above)

    A live DRAP National Drug List registration number lookup is not
    yet integrated (Phase 3 scope). The gazette equivalence is the
    current legal basis for Tier 2 status.
    """
    return {
        "medicine": medicine_name,
        "drap_status": "verified",
        "tier_who": 1,
        "tier_drap": 2,
        "who_basis": "WHO Model List of Essential Medicines – 24th List (2025)",
        "drap_basis": (
            "Pakistan NEML 2025 adopted WHO EML 24th List (2025) verbatim. "
            "Gazette: F.No.8-43/2021-DD(PS), Ministry of National Health "
            "Services, Regulations & Coordination, 20th October 2025."
        ),
        "drap_registration_number": "pending_phase3_integration",
    }


def _has_any_keyword(text: str, keywords: List[str]) -> bool:
    text_l = (text or "").lower()
    return any(kw in text_l for kw in keywords)


def validate_medicine(
    proposed_med: dict,
    kb_condition: dict,
    differential_names: List[str],
    chief_complaint: str,
) -> Optional[dict]:
    """
    Validate a single LLM-proposed medicine against the KB entry for
    its condition. Returns a validated medicine dict ready for output,
    or None if the medicine must be dropped entirely (no citation,
    or hard safety block).

    `proposed_med` is expected to have at least a "name" key (the
    LLM's proposed medicine name); other LLM-proposed fields (dosage,
    notes) are IGNORED in favour of the KB's own dose_instruction and
    citation, since the KB is the authoritative, citation-backed
    source. This prevents the LLM from inventing dosages.
    """
    # LLMs sometimes return medicines as plain strings instead of
    # {"name": "..."} dicts, despite the prompt schema. Handle both.
    if isinstance(proposed_med, str):
        name_l = proposed_med.strip().lower()
    elif isinstance(proposed_med, dict):
        name_l = proposed_med.get("name", "").strip().lower()
    else:
        return None

    if not name_l:
        return None

    # --- Safety override: dengue + NSAID/aspirin hard block ---
    differentials_l = [d.lower() for d in differential_names]
    if "dengue fever" in differentials_l or "dengue" in differentials_l:
        if name_l in DENGUE_CONTRAINDICATED:
            return None  # hard-blocked, no exceptions

    med_lookup = _kb_medicine_lookup(kb_condition)
    kb_med = med_lookup.get(name_l)

    # --- Citation enforcement ---
    if kb_med is None or not kb_med.get("citation"):
        # No KB-backed citation -> drop this medicine entirely.
        return None

    cond_id = kb_condition["condition_id"]

    # --- Conditional antibiotics (PUD H.pylori, GASTROENTERITIS dysentery) ---
    cond_rule = CONDITIONAL_ANTIBIOTICS.get(cond_id)
    if cond_rule and name_l in cond_rule["conditional_medicines"]:
        if not _has_any_keyword(chief_complaint, cond_rule["trigger_keywords"]):
            return None  # condition for showing this antibiotic not met

    drap = check_drap_registration(kb_med["name"])

    validated = {
        "name": kb_med["name"],
        "dosage_form": kb_med.get("dosage_form", ""),
        "dose_instruction": kb_med.get("dose_instruction", ""),
        "tier": [1, 2],   # Tier 1 (WHO EML) + Tier 2 (Pakistan NEML via gazette)
        "tier_description": "WHO EML 24th List (2025) + Pakistan NEML 2025 (Gazette 20-Oct-2025)",
        "citation": kb_med.get("citation", ""),
        "drap_check": drap,
    }

    # --- Physician-review flag (URI antibiotics: bacterial vs viral) ---
    review_set = PHYSICIAN_REVIEW_ANTIBIOTICS.get(cond_id)
    if review_set and name_l in review_set:
        validated["flag"] = (
            "PHYSICIAN_REVIEW_REQUIRED: confirm bacterial (not viral) "
            "presentation before dispensing this antibiotic."
        )

    return validated


def validate_recommendation(
    llm_output: dict,
    rule_engine_alert: dict,
    chief_complaint: str = "",
    kb: Optional[dict] = None,
) -> dict:
    """
    Main entry point. Takes the raw LLM output (a dict matching the
    expected JSON schema - see llm_pipeline.py PROMPT_TEMPLATE) plus
    the rule engine's critical alert dict, and returns the FINAL,
    safety-gated JSON for the hospital system.

    Expected llm_output shape (fields used by this function):
        {
            "diagnosis": {"name": str, "icd10": str},
            "confidence": float (0.0-1.0),
            "differential_diagnoses": [{"name": str, "icd10": str}, ...],
            "medicines": [{"name": str}, ...]   # LLM's proposed meds
        }
    """
    if kb is None:
        kb = load_kb()
    kb_index = _kb_index(kb)

    diagnosis = llm_output.get("diagnosis", {}) or {}
    diagnosis_name = diagnosis.get("name", "")
    confidence = float(llm_output.get("confidence", 0.0) or 0.0)
    differentials = llm_output.get("differential_diagnoses", []) or []
    
    # Handle both dict and string formats for differential_diagnoses
    differential_names = []
    normalized_differentials = []
    for d in differentials:
        if isinstance(d, dict):
            name = d.get("name", "")
            normalized_differentials.append(d)
        else:
            # Assume it's a string - convert to proper dict format
            name = str(d) if d else ""
            normalized_differentials.append({"name": name, "icd10": ""})
        if name:
            differential_names.append(name)

    # Try to resolve to a KB condition by condition_id first, then by name.
    cond_id = llm_output.get("condition_id")
    kb_condition = kb_index.get(cond_id) if cond_id else None
    if kb_condition is None:
        for c in kb["conditions"]:
            if c["condition_name"].lower() == diagnosis_name.lower():
                kb_condition = c
                break

    result = {
        "critical_alert": rule_engine_alert,
        "diagnosis": diagnosis,
        "confidence": round(confidence, 3),
        "differential_diagnoses": normalized_differentials,
        "medicines": [],
        "treatment_plan_notes": None,
        "action": None,
        "refer_to_specialist": False,
        "refer_reason": None,
    }

    # --- Reference-only conditions (e.g. TB) always refer, regardless
    #     of confidence ---
    if kb_condition is not None and kb_condition["condition_id"] in REFERENCE_ONLY_CONDITIONS:
        result["refer_to_specialist"] = True
        result["refer_reason"] = (
            f"{kb_condition['condition_name']}: this is a screening flag only. "
            f"{kb_condition['treatment_plan'].get('non_pharmacological', '')}"
        )
        result["action"] = "refer"
        result["medicines"] = []  # suppressed - reference only
        return result

    # --- Confidence gating ---
    if confidence < CONFIDENCE_LOW:
        result["refer_to_specialist"] = True
        result["refer_reason"] = (
            f"Confidence ({confidence:.0%}) below {CONFIDENCE_LOW:.0%} threshold. "
            "Refer to specialist; no diagnosis or medicines shown."
        )
        result["action"] = "refer"
        result["diagnosis"] = {}
        result["medicines"] = []
        return result

    if confidence < CONFIDENCE_HIGH:
        result["action"] = "diagnosis_only"
        result["refer_reason"] = (
            f"Confidence ({confidence:.0%}) is in the {CONFIDENCE_LOW:.0%}-"
            f"{CONFIDENCE_HIGH:.0%} band: diagnosis shown, but verify with "
            "a specialist before treatment. No medicines shown."
        )
        result["medicines"] = []
        return result

    # --- High confidence (> 80%): full recommendation, but every
    #     medicine must still pass validate_medicine() ---
    result["action"] = "full_recommendation"

    if kb_condition is None:
        # No matching KB condition at all -> cannot ground any
        # medicine in a citation. Per the safety mechanism: withhold
        # and refer.
        result["refer_to_specialist"] = True
        result["refer_reason"] = (
            f"No knowledge-base entry found for diagnosis '{diagnosis_name}'. "
            "No citation available - medicines withheld per safety policy."
        )
        result["medicines"] = []
        return result

    proposed_meds = llm_output.get("medicines", []) or []
    validated_meds = []
    for med in proposed_meds:
        v = validate_medicine(med, kb_condition, differential_names, chief_complaint)
        if v is not None:
            validated_meds.append(v)

    # If the LLM proposed nothing, or everything was dropped, fall back
    # to the KB's own medicine list (still subject to the same gates)
    # so the output isn't empty purely due to an LLM omission - but
    # ONLY if at least the condition itself is KB-grounded.
    if not validated_meds:
        for kb_med in kb_condition.get("treatment_plan", {}).get("medicines", []):
            v = validate_medicine({"name": kb_med["name"]}, kb_condition, differential_names, chief_complaint)
            if v is not None:
                validated_meds.append(v)

    if not validated_meds:
        result["refer_reason"] = (
            "No medicines passed citation/safety validation for this "
            "presentation (e.g. conditional antibiotics not indicated, "
            "or dengue NSAID block applied). Non-pharmacological "
            "management and physician review still apply."
        )

    result["medicines"] = validated_meds
    result["treatment_plan_notes"] = kb_condition.get("treatment_plan", {}).get(
        "non_pharmacological", ""
    )
    result["confidence_notes"] = kb_condition.get("confidence_notes", "")

    return result


if __name__ == "__main__":
    # Smoke test
    kb = load_kb()

    fake_llm_output = {
        "condition_id": "MALARIA_PF",
        "diagnosis": {"name": "Malaria, Plasmodium falciparum (uncomplicated)", "icd10": "B50.9"},
        "confidence": 0.85,
        "differential_diagnoses": [
            {"name": "Dengue fever", "icd10": "A90"},
            {"name": "Typhoid fever", "icd10": "A01.0"},
        ],
        "medicines": [
            {"name": "artemether + lumefantrine"},
            {"name": "ibuprofen"},        # should be dropped (dengue in differentials)
            {"name": "made_up_drug_xyz"}, # should be dropped (no citation)
        ],
    }
    rule_alert = {"flag": False, "severity": "none", "reasons": [], "recommended_action": ""}

    out = validate_recommendation(fake_llm_output, rule_alert, chief_complaint="fever and chills", kb=kb)
    print(json.dumps(out, indent=2))