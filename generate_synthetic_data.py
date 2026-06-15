"""
generate_synthetic_data.py
==========================
Generates clinically accurate synthetic patient vitals for all 15
conditions in the CDSS knowledge base.

Vital sign ranges are based on:
- WHO clinical guidelines
- Harrison's Principles of Internal Medicine
- Pakistan clinical context (tropical disease prevalence)

Each condition has its own realistic distribution of:
    bp_systolic, bp_diastolic, blood_sugar, weight_kg, height_cm,
    bmi (computed), temperature_c, pulse_bpm, age, sex

Usage:
    python generate_synthetic_data.py
        → writes data/synthetic_training_data.csv
        → writes data/synthetic_data_summary.json

    python generate_synthetic_data.py --rows 2000
        → 2000 rows per condition (default: 1500)

    python generate_synthetic_data.py --merge
        → also merges with existing classifier training data
          and writes data/merged_training_data.csv
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Helper: clipped normal distribution ───────────────────────────────────────

def cn(mean, sd, low, high, n):
    """Clipped normal — generates n values, clipped to [low, high]."""
    return np.clip(np.random.normal(mean, sd, n), low, high)


def choice(options, weights, n):
    """Weighted random choice."""
    return np.random.choice(options, size=n, p=weights)


def bmi_from(weight, height_cm):
    h_m = height_cm / 100.0
    return np.round(weight / (h_m ** 2), 1)


# ══════════════════════════════════════════════════════════════════════════════
# CONDITION GENERATORS
# Each function returns a DataFrame of n rows with label column set.
# Clinical ranges are evidence-based and Pakistan-context appropriate.
# ══════════════════════════════════════════════════════════════════════════════

def gen_hypertension(n):
    """
    Essential hypertension (I10).
    High BP is the defining feature. Sugar often elevated (metabolic syndrome).
    Pulse elevated due to cardiac strain. Often overweight/obese.
    Age skewed older (40+). More common in males in Pakistani population.
    """
    age    = cn(55, 12, 35, 80, n).astype(int)
    sex    = choice([1, 0], [0.55, 0.45], n)           # M slightly more
    height = cn(165, 8, 148, 185, n)
    weight = cn(82, 14, 55, 120, n)                     # overweight skew
    bp_s   = cn(162, 18, 140, 210, n)                   # high systolic
    bp_d   = cn(100, 10, 85, 130, n)                    # high diastolic
    sugar  = cn(115, 30, 80, 220, n)                    # often elevated
    temp   = cn(36.9, 0.3, 36.2, 37.5, n)
    pulse  = cn(85, 12, 60, 115, n)
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Hypertension")


def gen_type2_diabetes(n):
    """
    Type 2 Diabetes Mellitus (E11).
    High blood sugar is the cardinal feature. Often hypertensive too.
    Overweight/obese. Older age group. Elevated pulse (autonomic neuropathy).
    """
    age    = cn(50, 12, 30, 75, n).astype(int)
    sex    = choice([1, 0], [0.50, 0.50], n)
    height = cn(163, 8, 148, 182, n)
    weight = cn(85, 15, 55, 130, n)                     # obese skew
    bp_s   = cn(138, 18, 110, 185, n)                   # often hypertensive
    bp_d   = cn(86, 10, 68, 110, n)
    sugar  = cn(230, 60, 140, 450, n)                   # HIGH — defining feature
    temp   = cn(36.9, 0.3, 36.2, 37.5, n)
    pulse  = cn(84, 12, 62, 110, n)
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Type 2 Diabetes")


def gen_malaria(n):
    """
    Malaria P. falciparum (B50).
    HIGH fever (cyclical), elevated pulse, LOW BP (vasodilation),
    LOW-NORMAL sugar (parasite consumes glucose), normal-low weight.
    Young-to-middle age, male dominant in Pakistan (outdoor exposure).
    Temperature is the KEY differentiator from typhoid — spikes higher.
    """
    age    = cn(28, 12, 5, 60, n).astype(int)
    sex    = choice([1, 0], [0.65, 0.35], n)            # male outdoor exposure
    height = cn(165, 8, 148, 182, n)
    weight = cn(60, 10, 38, 88, n)                      # often thin
    bp_s   = cn(102, 12, 80, 130, n)                    # low (vasodilation)
    bp_d   = cn(65, 8,  50, 85,  n)
    sugar  = cn(82, 15, 50, 120, n)                     # low-normal
    temp   = cn(39.5, 0.7, 38.5, 41.5, n)              # HIGH fever, cyclical spikes
    pulse  = cn(112, 14, 88, 145, n)                    # tachycardia
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Malaria")


def gen_typhoid(n):
    """
    Typhoid fever / Enteric fever (A01.0).
    STEPWISE fever (lower than malaria, rises daily), RELATIVE BRADYCARDIA
    (pulse lower than expected for temp — Faget's sign), low BP.
    Young adults, both sexes equally. Contaminated water/food source.
    KEY differentiator: pulse LOWER relative to temperature vs malaria.
    """
    age    = cn(24, 10, 5, 50, n).astype(int)
    sex    = choice([1, 0], [0.52, 0.48], n)
    height = cn(163, 8, 148, 180, n)
    weight = cn(57, 10, 35, 85, n)
    bp_s   = cn(108, 12, 85, 135, n)                   # low
    bp_d   = cn(68, 8,  50, 88,  n)
    sugar  = cn(88, 14, 60, 120, n)
    temp   = cn(38.9, 0.6, 38.0, 40.5, n)             # lower than malaria, stepwise
    pulse  = cn(76, 10, 55, 100, n)                    # RELATIVE BRADYCARDIA — key!
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Typhoid")


def gen_uri_pharyngitis(n):
    """
    Upper Respiratory Infection / Pharyngitis (J06.9).
    Mild fever, mild tachycardia. ALL ages, peaks in children.
    Normal BP, normal sugar. Very common — high prevalence in dataset.
    """
    age    = cn(22, 16, 2, 65, n).astype(int)          # all ages, peak young
    sex    = choice([1, 0], [0.50, 0.50], n)
    height = cn(158, 12, 100, 182, n)                  # wide range (includes children)
    weight = cn(58, 15, 18, 95, n)
    bp_s   = cn(115, 12, 90, 145, n)                   # normal
    bp_d   = cn(74, 8,  55, 92,  n)
    sugar  = cn(92, 12, 70, 120, n)
    temp   = cn(37.8, 0.6, 37.2, 39.5, n)             # mild-moderate fever
    pulse  = cn(88, 12, 65, 118, n)                    # mild tachycardia
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "URI/Pharyngitis")


def gen_pneumonia(n):
    """
    Community-Acquired Pneumonia (J18.9).
    HIGH fever, significant tachycardia, lower BP (sepsis risk).
    Older age or very young. Elevated pulse strongly.
    """
    age    = cn(45, 20, 1, 80, n).astype(int)          # bimodal: elderly + infants
    sex    = choice([1, 0], [0.55, 0.45], n)
    height = cn(163, 10, 100, 182, n)
    weight = cn(65, 14, 20, 100, n)
    bp_s   = cn(112, 14, 82, 145, n)                   # low-normal (sepsis risk)
    bp_d   = cn(70, 9,  50, 92,  n)
    sugar  = cn(108, 25, 75, 180, n)                   # stress hyperglycaemia
    temp   = cn(39.0, 0.7, 38.0, 41.0, n)             # high fever
    pulse  = cn(110, 14, 88, 145, n)                   # significant tachycardia
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Pneumonia")


def gen_gastroenteritis(n):
    """
    Acute Gastroenteritis (A09).
    Low-normal BP (dehydration), elevated pulse (dehydration tachycardia),
    low-normal sugar (poor oral intake), low-grade or no fever.
    All ages. Very common in Pakistan (contaminated water/food).
    """
    age    = cn(25, 18, 1, 65, n).astype(int)
    sex    = choice([1, 0], [0.50, 0.50], n)
    height = cn(160, 12, 100, 182, n)
    weight = cn(58, 14, 18, 95, n)
    bp_s   = cn(106, 12, 82, 132, n)                   # LOW (dehydration)
    bp_d   = cn(67, 8,  48, 88,  n)
    sugar  = cn(85, 18, 55, 130, n)                    # low (poor intake)
    temp   = cn(37.5, 0.6, 36.8, 39.2, n)             # low-grade or no fever
    pulse  = cn(98, 14, 75, 132, n)                    # tachycardia (dehydration)
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Gastroenteritis")


def gen_anaemia(n):
    """
    Iron Deficiency Anaemia (D50).
    Low BP (reduced O2 carrying capacity), elevated pulse (compensatory),
    normal temperature, normal sugar. Female dominant (menstrual losses).
    Young women most affected in Pakistan.
    """
    age    = cn(28, 12, 10, 60, n).astype(int)
    sex    = choice([1, 0], [0.25, 0.75], n)            # strongly female
    height = cn(156, 7, 140, 175, n)
    weight = cn(50, 10, 32, 78, n)                      # often underweight
    bp_s   = cn(102, 10, 82, 125, n)                   # low
    bp_d   = cn(64, 7,  48, 82,  n)
    sugar  = cn(88, 14, 65, 118, n)
    temp   = cn(36.8, 0.3, 36.2, 37.4, n)             # afebrile
    pulse  = cn(98, 12, 75, 128, n)                    # compensatory tachycardia
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Anaemia")


def gen_uti(n):
    """
    Urinary Tract Infection (N39.0).
    Low-grade fever, mildly elevated pulse. Normal BP, normal sugar.
    Strongly female (anatomy). Reproductive age primarily.
    """
    age    = cn(30, 14, 15, 65, n).astype(int)
    sex    = choice([1, 0], [0.15, 0.85], n)            # strongly female
    height = cn(157, 7, 142, 175, n)
    weight = cn(58, 10, 38, 88, n)
    bp_s   = cn(115, 10, 92, 140, n)                   # normal
    bp_d   = cn(74, 7,  55, 92,  n)
    sugar  = cn(95, 18, 68, 145, n)                    # can be elevated in diabetics
    temp   = cn(37.8, 0.6, 37.0, 39.5, n)             # low-grade fever
    pulse  = cn(84, 10, 64, 108, n)                    # mildly elevated
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "UTI")


def gen_asthma(n):
    """
    Asthma (J45).
    Normal temperature (unless infective trigger), elevated pulse
    (bronchospasm + anxiety), normal-low BP, normal sugar.
    Young adults, both sexes. Atopic history.
    """
    age    = cn(28, 14, 5, 65, n).astype(int)
    sex    = choice([1, 0], [0.48, 0.52], n)            # slight female predominance
    height = cn(162, 9, 140, 182, n)
    weight = cn(66, 13, 38, 105, n)
    bp_s   = cn(118, 12, 92, 148, n)                   # normal to slightly low
    bp_d   = cn(75, 8,  55, 95,  n)
    sugar  = cn(95, 15, 68, 130, n)
    temp   = cn(37.2, 0.5, 36.5, 38.8, n)             # usually afebrile
    pulse  = cn(112, 14, 85, 148, n)                   # elevated (bronchospasm)
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Asthma")


def gen_dengue(n):
    """
    Dengue Fever (A90).
    HIGH fever, significant tachycardia, LOW BP (plasma leakage),
    normal-low sugar. Retro-orbital headache, rash (not captured in vitals).
    Young adults, urban Pakistan (Aedes mosquito, standing water).
    KEY differentiators vs malaria: urban setting, no cyclical fever,
    no rigors pattern, BP drops more significantly.
    """
    age    = cn(26, 10, 5, 55, n).astype(int)
    sex    = choice([1, 0], [0.58, 0.42], n)
    height = cn(164, 8, 148, 182, n)
    weight = cn(60, 10, 38, 88, n)
    bp_s   = cn(98, 12, 75, 125, n)                    # LOW (plasma leakage)
    bp_d   = cn(62, 8,  45, 82,  n)
    sugar  = cn(84, 14, 58, 115, n)                    # low-normal
    temp   = cn(39.2, 0.6, 38.2, 41.0, n)             # high, sustained (not cyclical)
    pulse  = cn(108, 14, 85, 142, n)                   # tachycardia
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Dengue")


def gen_tb_screening(n):
    """
    TB Screening / Suspected Pulmonary TB (Z03.8 / A15).
    LOW-GRADE prolonged fever, significant weight loss (low BMI),
    normal-low BP, elevated pulse. Older males, crowded living conditions.
    Chronic presentation — vitals reflect weeks/months of illness.
    """
    age    = cn(38, 14, 15, 70, n).astype(int)
    sex    = choice([1, 0], [0.65, 0.35], n)            # male predominant
    height = cn(165, 8, 150, 182, n)
    weight = cn(48, 10, 30, 72, n)                      # UNDERWEIGHT — weight loss
    bp_s   = cn(108, 12, 85, 135, n)                   # low-normal
    bp_d   = cn(68, 8,  50, 88,  n)
    sugar  = cn(88, 15, 62, 120, n)
    temp   = cn(37.8, 0.5, 37.2, 39.2, n)             # LOW-GRADE prolonged
    pulse  = cn(92, 12, 70, 118, n)                    # mildly elevated
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "TB Screening")


def gen_peptic_ulcer(n):
    """
    Peptic Ulcer Disease (K27).
    NORMAL vitals mostly — diagnosis is symptom-based (epigastric pain).
    Slightly elevated pulse (pain response). Normal temp, normal sugar.
    Middle-aged males, NSAID use, H. pylori. Stress, spicy food (Pakistan).
    """
    age    = cn(40, 12, 20, 68, n).astype(int)
    sex    = choice([1, 0], [0.62, 0.38], n)            # male predominant
    height = cn(166, 8, 150, 183, n)
    weight = cn(70, 12, 45, 105, n)
    bp_s   = cn(122, 12, 98, 152, n)                   # normal
    bp_d   = cn(78, 8,  58, 98,  n)
    sugar  = cn(98, 18, 70, 145, n)
    temp   = cn(37.0, 0.3, 36.4, 37.8, n)             # afebrile
    pulse  = cn(80, 10, 60, 104, n)                    # normal to mildly elevated
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Peptic Ulcer")


def gen_skin_infection(n):
    """
    Skin Infection / Cellulitis (L03).
    Low-to-moderate fever (local infection spreading), mildly elevated
    pulse, normal-mildly elevated BP. All ages, slight male predominance
    (trauma/outdoor wounds). Diabetics overrepresented.
    """
    age    = cn(38, 16, 5, 72, n).astype(int)
    sex    = choice([1, 0], [0.58, 0.42], n)
    height = cn(163, 9, 140, 183, n)
    weight = cn(72, 14, 40, 110, n)
    bp_s   = cn(124, 14, 95, 158, n)                   # normal to mildly elevated
    bp_d   = cn(78, 9,  58, 100, n)
    sugar  = cn(112, 35, 70, 260, n)                   # elevated (diabetics at risk)
    temp   = cn(38.1, 0.6, 37.2, 39.8, n)             # moderate fever
    pulse  = cn(88, 12, 65, 115, n)                    # mildly elevated
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Skin Infection")


def gen_conjunctivitis(n):
    """
    Acute Bacterial Conjunctivitis (H10).
    NORMAL vitals — this is a local eye infection. No systemic fever,
    no BP change, no pulse change. Diagnosis entirely symptom-based.
    All ages, equal sex. Common in children, schools, crowded settings.
    Vitals are essentially healthy baseline — classifier will struggle
    here (as expected — refer to specialist for eye cases).
    """
    age    = cn(22, 16, 2, 65, n).astype(int)
    sex    = choice([1, 0], [0.50, 0.50], n)
    height = cn(160, 12, 100, 182, n)
    weight = cn(60, 14, 18, 95, n)
    bp_s   = cn(115, 10, 92, 138, n)                   # normal
    bp_d   = cn(74, 7,  55, 92,  n)
    sugar  = cn(92, 12, 70, 115, n)
    temp   = cn(36.9, 0.3, 36.3, 37.5, n)             # afebrile — local infection
    pulse  = cn(76, 10, 58, 98,  n)                    # normal
    return _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, "Conjunctivitis")


# ── Builder ───────────────────────────────────────────────────────────────────

def _build(n, age, sex, height, weight, bp_s, bp_d, sugar, temp, pulse, label):
    """Assemble a DataFrame row with computed BMI and label."""
    weight = np.round(weight, 1)
    height = np.round(height, 1)
    bmi    = np.round(weight / ((height / 100) ** 2), 1)
    return pd.DataFrame({
        "bp_systolic":   np.round(bp_s, 0).astype(int),
        "bp_diastolic":  np.round(bp_d, 0).astype(int),
        "blood_sugar":   np.round(sugar, 1),
        "weight_kg":     weight,
        "height_cm":     height,
        "bmi":           bmi,
        "temperature_c": np.round(temp, 1),
        "pulse_bpm":     np.round(pulse, 0).astype(int),
        "age":           np.clip(age, 1, 95),
        "sex":           sex,
        "label":         label,
    })


# ── Registry: all 15 generators ───────────────────────────────────────────────

GENERATORS = {
    "Hypertension":    gen_hypertension,
    "Type 2 Diabetes": gen_type2_diabetes,
    "Malaria":         gen_malaria,
    "Typhoid":         gen_typhoid,
    "URI/Pharyngitis": gen_uri_pharyngitis,
    "Pneumonia":       gen_pneumonia,
    "Gastroenteritis": gen_gastroenteritis,
    "Anaemia":         gen_anaemia,
    "UTI":             gen_uti,
    "Asthma":          gen_asthma,
    "Dengue":          gen_dengue,
    "TB Screening":    gen_tb_screening,
    "Peptic Ulcer":    gen_peptic_ulcer,
    "Skin Infection":  gen_skin_infection,
    "Conjunctivitis":  gen_conjunctivitis,
}


# ── Validation: sanity check generated data ───────────────────────────────────

def validate_dataframe(df: pd.DataFrame, label: str) -> list:
    """Check for obvious clinical implausibilities."""
    issues = []
    # BMI range
    bad_bmi = df[(df["bmi"] < 10) | (df["bmi"] > 70)]
    if len(bad_bmi):
        issues.append(f"  {len(bad_bmi)} rows with implausible BMI")
    # Temperature
    bad_temp = df[(df["temperature_c"] < 35) | (df["temperature_c"] > 42)]
    if len(bad_temp):
        issues.append(f"  {len(bad_temp)} rows with implausible temperature")
    # Pulse
    bad_pulse = df[(df["pulse_bpm"] < 30) | (df["pulse_bpm"] > 200)]
    if len(bad_pulse):
        issues.append(f"  {len(bad_pulse)} rows with implausible pulse")
    # BP
    bad_bp = df[(df["bp_systolic"] < 60) | (df["bp_systolic"] > 250)]
    if len(bad_bp):
        issues.append(f"  {len(bad_bp)} rows with implausible BP")
    return issues


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all(rows_per_condition: int = 1500) -> pd.DataFrame:
    print(f"\nGenerating {rows_per_condition} rows × {len(GENERATORS)} conditions "
          f"= {rows_per_condition * len(GENERATORS):,} total rows\n")

    frames = []
    summary = {}

    for label, gen_fn in GENERATORS.items():
        df = gen_fn(rows_per_condition)

        issues = validate_dataframe(df, label)
        status = "OK" if not issues else f"WARNINGS: {'; '.join(issues)}"
        print(f"  {label:<22} {len(df):>5} rows  {status}")

        frames.append(df)
        summary[label] = {
            "rows":              len(df),
            "bp_systolic_mean":  round(df["bp_systolic"].mean(), 1),
            "temperature_mean":  round(df["temperature_c"].mean(), 1),
            "pulse_mean":        round(df["pulse_bpm"].mean(), 1),
            "blood_sugar_mean":  round(df["blood_sugar"].mean(), 1),
            "bmi_mean":          round(df["bmi"].mean(), 1),
            "age_mean":          round(df["age"].mean(), 1),
            "pct_male":          round(df["sex"].mean() * 100, 1),
        }

    combined = pd.concat(frames, ignore_index=True)

    # Shuffle so conditions aren't in blocks
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nTotal rows: {len(combined):,}")
    print(f"Class distribution:\n{combined['label'].value_counts().to_string()}")

    return combined, summary


def merge_with_existing(synthetic_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Look for existing training CSVs and merge with synthetic data.
    Expected location: data/existing_training_data.csv
    The CSV must have the same columns as synthetic_df.
    """
    existing_path = os.path.join(DATA_DIR, "existing_training_data.csv")
    if not os.path.exists(existing_path):
        print(f"\n[merge] No existing data found at {existing_path}")
        print("[merge] Place your merged Kaggle CSV there to combine.")
        return None

    existing = pd.read_csv(existing_path)
    print(f"\n[merge] Found existing data: {len(existing):,} rows")

    # Keep only columns that match
    common_cols = [c for c in synthetic_df.columns if c in existing.columns]
    if "label" not in common_cols:
        print("[merge] ERROR: existing data has no 'label' column — skipping merge.")
        return None

    merged = pd.concat([existing[common_cols], synthetic_df[common_cols]],
                       ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[merge] Merged total: {len(merged):,} rows")
    print(f"[merge] Class distribution:\n{merged['label'].value_counts().to_string()}")
    return merged


def print_retraining_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           NEXT STEP: RETRAIN YOUR XGBOOST CLASSIFIER            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Open mergingdata.py (your Colab training script)            ║
║                                                                  ║
║  2. Add this at the top of the data loading section:            ║
║                                                                  ║
║     synthetic = pd.read_csv("data/synthetic_training_data.csv") ║
║     df = pd.concat([df, synthetic], ignore_index=True)          ║
║                                                                  ║
║  3. Re-run the XGBoost training                                  ║
║                                                                  ║
║  4. Replace:                                                     ║
║     data/classifier.pkl                                          ║
║     data/label_encoder.pkl                                       ║
║                                                                  ║
║  5. Run: python evaluate_pipeline.py                             ║
║     Expected improvement: 65% → 80-88% accuracy                 ║
║                                                                  ║
║  If you also have Kaggle CSVs to add:                           ║
║     python generate_synthetic_data.py --merge                   ║
║     → Put your Kaggle CSV at data/existing_training_data.csv    ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDSS Synthetic Data Generator")
    parser.add_argument("--rows",  type=int, default=1500,
                        help="Rows per condition (default: 1500)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing data/existing_training_data.csv")
    parser.add_argument("--out",   default="data/synthetic_training_data.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    synthetic_df, summary = generate_all(args.rows)

    # Save synthetic CSV
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    synthetic_df.to_csv(out_path, index=False)
    print(f"\nSynthetic data saved -> {out_path}")

    # Save summary JSON
    summary_meta = {
        "generated_at":       datetime.now().isoformat(),
        "rows_per_condition": args.rows,
        "total_rows":         len(synthetic_df),
        "conditions":         len(GENERATORS),
        "features":           [c for c in synthetic_df.columns if c != "label"],
        "clinical_notes": {
            "Malaria_vs_Typhoid": (
                "Key differentiator: Malaria temp mean ~39.5C, pulse ~112. "
                "Typhoid temp mean ~38.9C, pulse ~76 (relative bradycardia). "
                "This should fix the typhoid over-diagnosis in the classifier."
            ),
            "Dengue_vs_Malaria": (
                "Dengue: lower BP (plasma leakage), urban setting. "
                "Malaria: higher temp spikes, rural travel."
            ),
            "Conjunctivitis": (
                "Vitals are near-normal — classifier will always struggle here. "
                "This condition is best routed via chief complaint (RAG), not vitals."
            ),
            "Anaemia": (
                "Strongly female (75%). Low BP + compensatory tachycardia pattern."
            ),
        },
        "condition_stats": summary,
    }

    summary_path = os.path.join(DATA_DIR, "synthetic_data_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_meta, f, indent=2)
    print(f"Summary saved      -> {summary_path}")

    # Merge if requested
    if args.merge:
        merged_df = merge_with_existing(synthetic_df)
        if merged_df is not None:
            merged_path = os.path.join(DATA_DIR, "merged_training_data.csv")
            merged_df.to_csv(merged_path, index=False)
            print(f"Merged data saved  -> {merged_path}")

    print_retraining_instructions()
