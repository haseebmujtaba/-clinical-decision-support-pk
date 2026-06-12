"""
rule_engine.py
==============
Phase 2 - Step 1: Rule engine (pure Python, no ML/LLM dependency).

Runs FIRST in the pipeline, on raw vitals, before the classifier or
LLM ever see the data. Its job is purely to catch life-threatening
values and raise a CRITICAL ALERT flag.

This is a patient-safety mechanism, not a diagnostic tool. If
critical_alert.flag == True, the calling system (main.py) should:
    - still run the rest of the pipeline (for documentation), but
    - prioritise the critical_alert block in the output JSON, and
    - the hospital UI should surface this immediately to a physician
      regardless of the AI's diagnosis/confidence.

Thresholds are based on standard adult emergency-medicine cut-offs
(WHO Emergency Triage Assessment and Treatment - ETAT, and common
hospital protocol thresholds). These thresholds are Tier4
(hospital-protocol level) and are NOT meant to replace local
hospital emergency protocols - they should be reviewed/approved by
the hospital's clinical governance committee before production use.

NOTE ON PAEDIATRIC PATIENTS:
The thresholds below are calibrated for ADULTS (age >= 18). For
age < 18, several thresholds (esp. pulse, BP, temperature) differ
by age. This Phase-2 rule engine flags any patient < 18 with a
SEPARATE_PAEDIATRIC_PROTOCOL_REQUIRED warning so adult thresholds
are never silently misapplied to a child. A full paediatric ruleset
is out of scope for Phase 2.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VitalsInput:
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    blood_sugar: Optional[float] = None          # mg/dL
    blood_sugar_context: str = "random"          # "fasting" | "random" | "post_meal"
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    temperature_c: Optional[float] = None
    pulse_bpm: Optional[float] = None
    age: Optional[int] = None
    sex: Optional[str] = None                    # "M" | "F"
    chief_complaint: str = ""


@dataclass
class CriticalAlert:
    flag: bool = False
    severity: str = "none"            # "none" | "warning" | "critical" | "emergency"
    reasons: List[str] = field(default_factory=list)
    recommended_action: str = ""

    def to_dict(self):
        return {
            "flag": self.flag,
            "severity": self.severity,
            "reasons": self.reasons,
            "recommended_action": self.recommended_action,
        }


def _bmi(v: VitalsInput) -> Optional[float]:
    if v.weight_kg and v.height_cm:
        h_m = v.height_cm / 100.0
        if h_m > 0:
            return round(v.weight_kg / (h_m * h_m), 1)
    return None


# Each entry: (condition_fn, severity, message, action)
ADULT_RULES = [
    # --- Blood pressure ---
    (
        lambda v: v.bp_systolic is not None and v.bp_diastolic is not None
        and (v.bp_systolic >= 180 or v.bp_diastolic >= 120),
        "emergency",
        "Hypertensive crisis (BP >= 180/120 mmHg)",
        "Refer immediately for emergency evaluation of hypertensive crisis "
        "(possible hypertensive emergency/encephalopathy). Do not discharge.",
    ),
    (
        lambda v: v.bp_systolic is not None and v.bp_diastolic is not None
        and (v.bp_systolic < 90 or v.bp_diastolic < 60),
        "critical",
        "Hypotension (BP < 90/60 mmHg) - possible shock",
        "Assess for shock (sepsis, haemorrhage, cardiac, anaphylaxis). "
        "Urgent physician review required.",
    ),

    # --- Blood sugar ---
    (
        lambda v: v.blood_sugar is not None and v.blood_sugar < 54,
        "emergency",
        "Severe hypoglycaemia (blood sugar < 54 mg/dL)",
        "Administer fast-acting glucose immediately per hospital "
        "hypoglycaemia protocol. Recheck blood sugar after 15 minutes.",
    ),
    (
        lambda v: v.blood_sugar is not None and 54 <= v.blood_sugar < 70,
        "warning",
        "Hypoglycaemia (blood sugar 54-69 mg/dL)",
        "Give oral fast-acting carbohydrate if patient is conscious and "
        "able to swallow safely. Recheck blood sugar.",
    ),
    (
        lambda v: v.blood_sugar is not None and v.blood_sugar >= 400,
        "emergency",
        "Severe hyperglycaemia (blood sugar >= 400 mg/dL) - possible DKA/HHS",
        "Urgent assessment for diabetic ketoacidosis (DKA) or "
        "hyperosmolar hyperglycaemic state (HHS): check for ketones, "
        "dehydration, altered consciousness. Refer for emergency care.",
    ),
    (
        lambda v: v.blood_sugar is not None and 250 <= v.blood_sugar < 400,
        "warning",
        "Marked hyperglycaemia (blood sugar 250-399 mg/dL)",
        "Assess hydration and ketone status; physician review before "
        "starting/adjusting hypoglycaemic medicines.",
    ),

    # --- Temperature ---
    (
        lambda v: v.temperature_c is not None and v.temperature_c >= 39.5,
        "critical",
        "Hyperpyrexia (temperature >= 39.5 degC)",
        "Initiate active cooling and investigate source of fever "
        "urgently (consider sepsis, severe malaria, meningitis, "
        "dengue, typhoid depending on context). Physician review required.",
    ),
    (
        lambda v: v.temperature_c is not None and v.temperature_c <= 35.0,
        "critical",
        "Hypothermia (temperature <= 35.0 degC)",
        "Initiate active rewarming and assess for sepsis, exposure, or "
        "endocrine causes. Physician review required.",
    ),

    # --- Pulse ---
    (
        lambda v: v.pulse_bpm is not None and (v.pulse_bpm >= 130 or v.pulse_bpm <= 40),
        "critical",
        "Severe heart rate abnormality (pulse >= 130 or <= 40 bpm)",
        "Obtain ECG and urgent physician review for arrhythmia, "
        "shock, or conduction abnormality.",
    ),
    (
        lambda v: v.pulse_bpm is not None and (110 <= v.pulse_bpm < 130 or 41 <= v.pulse_bpm <= 49),
        "warning",
        "Heart rate outside normal range (pulse 110-129 or 41-49 bpm)",
        "Correlate with clinical context (fever, pain, dehydration, "
        "athletic conditioning); physician to assess if abnormal "
        "for this patient.",
    ),

    # --- BMI (informational, not emergency) ---
    (
        lambda v: _bmi(v) is not None and _bmi(v) >= 40,
        "warning",
        "BMI >= 40 (Class III obesity)",
        "Flag for nutrition/weight-management referral and assess for "
        "obesity-related comorbidities (T2DM, hypertension, OSA).",
    ),
    (
        lambda v: _bmi(v) is not None and _bmi(v) < 16,
        "warning",
        "BMI < 16 (severe underweight)",
        "Assess for malnutrition, chronic disease, or eating disorder; "
        "consider nutrition referral.",
    ),
]


def run_rule_engine(vitals: VitalsInput) -> CriticalAlert:
    """
    Run all rule-engine checks on the given vitals and return a
    CriticalAlert summary. This function MUST run before the
    classifier/LLM, and its output MUST be passed through to the
    final JSON regardless of what the AI layers decide.
    """
    alert = CriticalAlert()

    # Paediatric guard: adult thresholds are not valid for children.
    if vitals.age is not None and vitals.age < 18:
        alert.flag = True
        alert.severity = "warning"
        alert.reasons.append(
            "Patient is under 18 years old. Adult critical-value "
            "thresholds in this rule engine do NOT apply. "
            "Paediatric-specific thresholds are required "
            "(SEPARATE_PAEDIATRIC_PROTOCOL_REQUIRED, out of scope "
            "for Phase 2)."
        )
        alert.recommended_action = (
            "Route to a paediatric protocol / paediatrician. "
            "Do not rely on adult thresholds for this patient."
        )
        return alert

    severity_rank = {"none": 0, "warning": 1, "critical": 2, "emergency": 3}
    actions = []

    for condition_fn, severity, message, action in ADULT_RULES:
        try:
            if condition_fn(vitals):
                alert.flag = True
                alert.reasons.append(message)
                actions.append(action)
                if severity_rank[severity] > severity_rank[alert.severity]:
                    alert.severity = severity
        except Exception:
            # Defensive: missing/odd data should never crash the rule engine.
            continue

    if actions:
        alert.recommended_action = " | ".join(dict.fromkeys(actions))

    return alert


if __name__ == "__main__":
    test_cases = [
        VitalsInput(bp_systolic=190, bp_diastolic=125, blood_sugar=110,
                     weight_kg=70, height_cm=170, temperature_c=37.0,
                     pulse_bpm=88, age=55, sex="M",
                     chief_complaint="Severe headache"),
        VitalsInput(bp_systolic=120, bp_diastolic=80, blood_sugar=45,
                     weight_kg=60, height_cm=160, temperature_c=36.5,
                     pulse_bpm=100, age=30, sex="F",
                     chief_complaint="Dizziness, sweating"),
        VitalsInput(bp_systolic=110, bp_diastolic=70, blood_sugar=140,
                     weight_kg=65, height_cm=165, temperature_c=38.2,
                     pulse_bpm=92, age=28, sex="F",
                     chief_complaint="Fever and cough"),
        VitalsInput(bp_systolic=100, bp_diastolic=65, blood_sugar=130,
                     weight_kg=20, height_cm=110, temperature_c=39.8,
                     pulse_bpm=140, age=6, sex="M",
                     chief_complaint="High fever, child very sleepy"),
    ]
    for i, tc in enumerate(test_cases, 1):
        result = run_rule_engine(tc)
        print(f"\n--- Test case {i} ---")
        print(result.to_dict())