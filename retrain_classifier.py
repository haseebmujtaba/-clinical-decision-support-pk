"""
retrain_classifier.py
=====================
Retrains the XGBoost classifier on merged_training_data_expanded.csv.
Handles the data quality issues found in the CSV:
  - 85%+ missing values in most vital columns
  - 6400 garbage/corrupted diagnosis labels (numeric strings)
  - Severe class imbalance (467k No Diabetes vs tiny other classes)
  - Diagnosis label names that don't match the pipeline's KB conditions

Steps:
  1. Load CSV
  2. Clean: drop corrupted rows, normalize diagnosis names to KB labels
  3. Drop rows where ALL key vitals are missing
  4. Impute remaining missing values with column medians
  5. Balance classes (cap majority classes, oversample minority)
  6. Train XGBoost
  7. Evaluate + print classification report
  8. Save new classifier.pkl + label_encoder.pkl

Run:
    python retrain_classifier.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import xgboost as xgb
import joblib

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
INPUT_CSV  = os.path.join(DATA_DIR, "merged_training_data_expanded.csv")
OUT_CLF    = os.path.join(DATA_DIR, "classifier.pkl")
OUT_ENC    = os.path.join(DATA_DIR, "label_encoder.pkl")
OUT_CARD   = os.path.join(DATA_DIR, "training_data_card.json")

FEATURES = [
    "bp_systolic", "bp_diastolic", "blood_sugar",
    "weight_kg", "height_cm", "bmi",
    "temperature_c", "pulse_bpm", "age", "sex",
]

# ---------------------------------------------------------------------
# Diagnosis label normalization map
# Maps raw CSV diagnosis values → KB condition names used in pipeline
# ---------------------------------------------------------------------
LABEL_MAP = {
    # Diabetes-related
    "Type 2 Diabetes":          "Type 2 Diabetes",
    "Diabetes":                 "Type 2 Diabetes",
    "Pre-diabetes":             "Type 2 Diabetes",
    "No Diabetes":              "No Diabetes",

    # Hypertension-related
    "Hypertension":             "Hypertension",
    "No Hypertension":          "No Hypertension",
    "No Disease":               "No Hypertension",

    # Cardiovascular
    "Cardio Disease":           "Coronary Artery Disease",
    "Heart Disease":            "Coronary Artery Disease",
    "Coronary Artery Disease":  "Coronary Artery Disease",

    # Respiratory
    "Asthma":                   "Asthma",
    "Pneumonia":                "Pneumonia",
    "Bronchitis":               "Bronchitis",
    "Influenza":                "Influenza",
    "Common Cold":              "Common Cold",

    # Infectious
    "Malaria":                  "Malaria, Plasmodium falciparum (uncomplicated)",
    "Typhoid Fever":            "Typhoid fever (Enteric fever)",
    "Dengue Fever":             "Dengue fever",
    "Tuberculosis":             "Tuberculosis (pulmonary, presumptive)",
    "Urinary Tract Infection":  "Urinary Tract Infection (uncomplicated)",
    "UTI":                      "Urinary Tract Infection (uncomplicated)",
    "Gastroenteritis":          "Acute Gastroenteritis / Diarrhoea",

    # Other conditions
    "Anaemia":                  "Anaemia (iron-deficiency)",
    "Anemia":                   "Anaemia (iron-deficiency)",
    "Stroke":                   "Stroke",
    "Depression":               "Depression",
    "Anxiety Disorders":        "Anxiety Disorders",
    "Migraine":                 "Migraine",
    "Osteoporosis":             "Osteoporosis",
    "Osteoarthritis":           "Osteoarthritis",
    "Rheumatoid Arthritis":     "Rheumatoid Arthritis",
    "Kidney Disease":           "Kidney Disease",
    "Liver Disease":            "Liver Disease",
    "Hypothyroidism":           "Hypothyroidism",
    "Hyperthyroidism":          "Hyperthyroidism",
    "Alzheimer's Disease":      "Alzheimer's Disease",
    "Parkinson's Disease":      "Parkinson's Disease",
    "Multiple Sclerosis":       "Multiple Sclerosis",
    "Psoriasis":                "Psoriasis",
    "Eczema":                   "Eczema",
    "Peptic Ulcer":             "Peptic Ulcer Disease",
    "Crohn's Disease":          "Crohn's Disease",
    "Ulcerative Colitis":       "Ulcerative Colitis",
    "Pancreatitis":             "Pancreatitis",
    "Liver Cancer":             "Liver Cancer",
    "Kidney Cancer":            "Kidney Cancer",
    "Allergic Rhinitis":        "Allergic Rhinitis",
    "Conjunctivitis":           "Conjunctivitis (bacterial)",
}

# Minimum samples needed to keep a class (drop smaller classes)
MIN_SAMPLES = 50
# Maximum samples per class (cap to reduce imbalance)
MAX_SAMPLES = 5000


def load_and_clean(path: str) -> pd.DataFrame:
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # --- Step 1: Normalize sex column to numeric (M=1, F=0) ---
    if df["sex"].dtype == object:
        df["sex"] = df["sex"].map({"M": 1, "F": 0, "Male": 1, "Female": 0})
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce").fillna(0).astype(int)

    # --- Step 2: Drop corrupted diagnosis rows ---
    # Keep only rows whose diagnosis appears in our label map
    valid_labels = set(LABEL_MAP.keys())
    original_len = len(df)
    df = df[df["diagnosis"].isin(valid_labels)].copy()
    dropped = original_len - len(df)
    print(f"  Dropped {dropped:,} rows with corrupted/unknown diagnosis labels")
    print(f"  Remaining: {len(df):,} rows")

    # --- Step 3: Normalize diagnosis labels to KB names ---
    df["diagnosis"] = df["diagnosis"].map(LABEL_MAP)

    # --- Step 4: Drop rows where ALL key vitals are missing ---
    # A row with no vitals at all is useless for training
    key_vitals = ["bp_systolic", "bp_diastolic", "blood_sugar",
                  "temperature_c", "pulse_bpm"]
    before = len(df)
    df = df.dropna(subset=key_vitals, how="all")
    print(f"  Dropped {before - len(df):,} rows with ALL key vitals missing")
    print(f"  Remaining: {len(df):,} rows")

    # --- Step 5: Impute remaining missing values with column medians ---
    for col in FEATURES:
        if col in df.columns:
            median_val = df[col].median()
            missing = df[col].isnull().sum()
            if missing > 0:
                df[col] = df[col].fillna(median_val)

    # Recompute BMI where missing or zero using weight + height
    mask = (df["bmi"].isna() | (df["bmi"] <= 0)) & (df["height_cm"] > 0)
    df.loc[mask, "bmi"] = (
        df.loc[mask, "weight_kg"] /
        ((df.loc[mask, "height_cm"] / 100) ** 2)
    ).round(1)
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    return df


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    print("\nBalancing classes...")
    counts = df["diagnosis"].value_counts()
    print(f"  Classes before balancing: {len(counts)}")

    # Drop classes with too few samples
    valid_classes = counts[counts >= MIN_SAMPLES].index
    df = df[df["diagnosis"].isin(valid_classes)].copy()
    print(f"  Classes with >= {MIN_SAMPLES} samples: {len(valid_classes)}")

    # Cap majority classes and oversample minority classes
    balanced_dfs = []
    for label in valid_classes:
        class_df = df[df["diagnosis"] == label]
        n = len(class_df)

        if n > MAX_SAMPLES:
            # Downsample large classes
            class_df = resample(class_df, n_samples=MAX_SAMPLES,
                                replace=False, random_state=42)
        elif n < 200:
            # Oversample very small classes
            class_df = resample(class_df, n_samples=200,
                                replace=True, random_state=42)

        balanced_dfs.append(class_df)

    balanced = pd.concat(balanced_dfs, ignore_index=True)
    print(f"  Shape after balancing: {balanced.shape}")
    print("\n  Final class distribution:")
    counts_after = balanced["diagnosis"].value_counts()
    for label, count in counts_after.items():
        print(f"    {label}: {count:,}")
    return balanced


def train(df: pd.DataFrame):
    print("\nPreparing features and labels...")
    X = df[FEATURES].values
    y_raw = df["diagnosis"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"  Features: {X.shape}, Labels: {len(le.classes_)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    print("\nTraining XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall accuracy: {acc:.1%}")
    print("\n  Per-class report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    # Feature importance
    print("  Feature importance:")
    importances = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(importances.to_string(index=False))

    return model, le, acc, importances


def save_artifacts(model, le, acc, importances, df):
    print(f"\nSaving artifacts to {DATA_DIR}/...")
    joblib.dump(model, OUT_CLF)
    joblib.dump(le, OUT_ENC)
    print(f"  Saved classifier.pkl")
    print(f"  Saved label_encoder.pkl")

    card = {
        "version": "2.0",
        "trained_on": "merged_training_data_expanded.csv",
        "training_rows_after_cleaning": len(df),
        "n_classes": len(le.classes_),
        "classes": list(le.classes_),
        "overall_accuracy": round(acc, 4),
        "features": FEATURES,
        "sex_encoding": {"M": 1, "F": 0},
        "max_samples_per_class": MAX_SAMPLES,
        "min_samples_per_class": MIN_SAMPLES,
        "compliance_note": (
            "XGBoost classifier for CDSS Phase 2. Used as weak signal "
            "only — not ground truth. Final diagnosis gated by "
            "citation_validator.py confidence thresholds."
        ),
    }
    with open(OUT_CARD, "w") as f:
        json.dump(card, f, indent=2)
    print(f"  Saved training_data_card.json")
    print(f"\nDone. New classifier accuracy: {acc:.1%}")
    print("Run python evaluate_pipeline.py to test the full pipeline.")


def main():
    print("=" * 60)
    print("CDSS Phase 2 — XGBoost Classifier Retraining")
    print("=" * 60)

    df = load_and_clean(INPUT_CSV)
    df = balance_classes(df)
    model, le, acc, importances = train(df)
    save_artifacts(model, le, acc, importances, df)


if __name__ == "__main__":
    main()
