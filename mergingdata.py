import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================
# Define a structured directory for input/output files to keep VS Code clean
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

COMMON_COLUMNS = [
    "bp_systolic", "bp_diastolic", "blood_sugar", "weight_kg",
    "height_cm", "bmi", "temperature_c", "pulse_bpm",
    "age", "sex", "diagnosis", "source_dataset"
]

FEATURE_COLUMNS = [
    "bp_systolic", "bp_diastolic", "blood_sugar", "weight_kg", 
    "height_cm", "bmi", "temperature_c", "pulse_bpm", "age", "sex"
]

def get_path(filename):
    return os.path.join(DATA_DIR, filename)

def empty_common_df():
    return pd.DataFrame(columns=COMMON_COLUMNS)

# ============================================================
# DAY 3 — STANDARDIZE & MERGE DATASETS
# ============================================================

def standardize_diabetes(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing resource: {path}")
    
    df = pd.read_csv(path)
    out = empty_common_df()

    out["bp_diastolic"] = df["BloodPressure"]
    out["blood_sugar"]  = df["Glucose"]
    out["bmi"]          = df["BMI"]
    out["age"]          = df["Age"]
    out["sex"]          = 0  # 0 = female
    
    out["diagnosis"] = df["Outcome"].map({
        1: "Type 2 Diabetes",
        0: "No Diabetes"
    })
    out["source_dataset"] = "pima_diabetes"
    return out


def standardize_heart_disease(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing resource: {path}")
        
    df = pd.read_csv(path)
    out = empty_common_df()

    out["age"]          = df["age"]
    out["bp_systolic"]  = df["trestbps"]
    out["pulse_bpm"]    = df["thalch"]
    out["sex"]          = df["sex"].map({"Male": 1, "Female": 0})
    
    out["diagnosis"] = df["num"].apply(
        lambda x: "Hypertension" if x > 0 else "No Hypertension"
    )
    out["source_dataset"] = "heart_disease_uci"
    return out


def standardize_disease_symptom(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing resource: {path}")
        
    df = pd.read_csv(path)
    out = empty_common_df()

    out["age"] = df["Age"]
    out["sex"] = df["Gender"].map({"Male": 1, "Female": 0})

    bp_map = {"Low": 95, "Normal": 120, "High": 145}
    out["bp_systolic"] = df["Blood Pressure"].map(bp_map)
    out["diagnosis"] = df["Disease"]
    out["source_dataset"] = "disease_symptom_profile"
    return out

# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================
def main():
    # 1. Load and Standardize
    print("-" * 60)
    print("STEP 1: Standardizing and Merging Datasets...")
    print("-" * 60)
    
    try:
        df_diabetes = standardize_diabetes(get_path("diabetes.csv"))
        df_heart    = standardize_heart_disease(get_path("heart_disease_uci.csv"))
        # Using "Disease.csv" to perfectly match your physical file sidebar name
        df_symptom  = standardize_disease_symptom(get_path("Disease.csv"))
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print(f"Please ensure your CSV files are placed inside the '{DATA_DIR}/' folder in your VS Code workspace.")
        return

    merged = pd.concat([df_diabetes, df_heart, df_symptom], ignore_index=True)

    print(f" -> Pima diabetes rows:    {len(df_diabetes)}")
    print(f" -> Heart disease rows:    {len(df_heart)}")
    print(f" -> Disease/symptom rows:  {len(df_symptom)}")
    print(f"TOTAL merged rows:         {len(merged)}")

    # 2. Day 4: Clean Datasets
    print("\n" + "-" * 60)
    print("STEP 2: Cleaning Merged Data...")
    print("-" * 60)
    
    # Drop rows with missing diagnosis
    merged = merged.dropna(subset=["diagnosis"])
    
    # Fill missing numeric columns with median or clinical defaults
    defaults = {
        "bp_systolic": 120, "bp_diastolic": 80, "blood_sugar": 100,
        "weight_kg": 70, "height_cm": 165, "bmi": 24,
        "temperature_c": 37.0, "pulse_bpm": 75, "age": 40
    }

    for col in defaults.keys():
        median_val = merged[col].median()
        # Fallback if the entire column is NaN
        if pd.isna(median_val):
            median_val = defaults[col]
        merged[col] = merged[col].fillna(median_val)

    # Impute categorical sex column
    sex_mode = merged["sex"].mode()
    default_sex = sex_mode[0] if not sex_mode.empty else 0
    merged["sex"] = merged["sex"].fillna(default_sex).astype(int)

    # Recalculate BMI where both weight & height exist natively
    has_wh = merged["weight_kg"].notna() & merged["height_cm"].notna()
    if has_wh.any():
        merged.loc[has_wh, "bmi"] = (
            merged.loc[has_wh, "weight_kg"] /
            (merged.loc[has_wh, "height_cm"] / 100) ** 2
        )

    # Remove low-frequency diagnosis classes (<5)
    label_counts = merged["diagnosis"].value_counts()
    valid_labels = label_counts[label_counts >= 5].index
    merged = merged[merged["diagnosis"].isin(valid_labels)]
    print(f" -> Remaining samples after rare class filtering: {len(merged)}")

    # Save cleaned dataset
    cleaned_csv_path = get_path("merged_training_data.csv")
    merged.to_csv(cleaned_csv_path, index=False)
    print(f" -> Saved cleaned dataset to: {cleaned_csv_path}")

    # 3. Day 5: Train and Evaluate
    print("\n" + "-" * 60)
    print("STEP 3: Training XGBoost Classifier...")
    print("-" * 60)
    
    # CRITICAL FIX: Explicitly coerce all feature columns into float/numeric types.
    # This strips away residual text object representations that break XGBoost.
    print("Forcing feature columns into numerical representations...")
    for col in FEATURE_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
        # Clean up any fresh NaNs generated during mapping coercion
        col_median = merged[col].median()
        fallback_val = col_median if not pd.isna(col_median) else defaults.get(col, 0)
        merged[col] = merged[col].fillna(fallback_val)

    X = merged[FEATURE_COLUMNS]
    y = merged["diagnosis"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)
    print(" -> Model training complete.")

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n==================================================")
    print(f"OVERALL TEST ACCURACY: {accuracy*100:.2f}%")
    print(f"==================================================\n")
    print("Detailed Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

    # Feature Importance
    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    print("\nFeature Importance Profile:")
    print(importance_df.to_string(index=False))

    # Save Model Artifacts
    model_path = get_path("classifier.pkl")
    encoder_path = get_path("label_encoder.pkl")
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

    # 4. Generate Training Data Card
    data_card = {
        "model_name": "CDSS Vitals Classifier",
        "model_type": "XGBoost Multi-class Classifier",
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "total_training_samples": int(len(merged)),
        "train_test_split": "80/20 stratified",
        "test_accuracy": f"{accuracy*100:.2f}%",
        "features_used": FEATURE_COLUMNS,
        "diagnosis_classes": list(encoder.classes_),
        "datasets": [
            {"name": "Pima Indians Diabetes Database", "rows_contributed": int(len(df_diabetes))},
            {"name": "Heart Disease UCI Dataset", "rows_contributed": int(len(df_heart))},
            {"name": "Disease Symptom and Patient Profile Dataset", "rows_contributed": int(len(df_symptom))}
        ]
    }

    card_path = get_path("training_data_card.json")
    with open(card_path, "w") as f:
        json.dump(data_card, f, indent=2)

    print("\n" + "="*50)
    print("SUCCESS: Pipeline Run Complete. Artifacts generated:")
    print(f"  - {cleaned_csv_path}")
    print(f"  - {model_path}")
    print(f"  - {encoder_path}")
    print(f"  - {card_path}")
    print("="*50)

    # 5. Inference Sanity Check
    print("\nRunning Inference Validation...")
    sample_patient = pd.DataFrame([{
        "bp_systolic": 145, "bp_diastolic": 92, "blood_sugar": 210,
        "weight_kg": 82, "height_cm": 170, "bmi": 82 / (1.70 ** 2),
        "temperature_c": 37.1, "pulse_bpm": 88, "age": 52, "sex": 1
    }])
    
    pred_encoded = model.predict(sample_patient)
    pred_proba = model.predict_proba(sample_patient)
    pred_label = encoder.inverse_transform(pred_encoded)

    print(f" -> Sample Prediction Output: [ {pred_label[0]} ]")
    proba_df = pd.DataFrame({
        "Diagnosis": encoder.classes_,
        "Probability": pred_proba[0]
    }).sort_values("Probability", ascending=False).head(3)
    print("\nTop 3 Probabilities:")
    print(proba_df.to_string(index=False))

if __name__ == "__main__":
    main()