"""Interactive local tester for NAVYA's exported cycle-length model.

Run after Notebook 10 has exported models/cycle_length_model.joblib and metadata.
This script is for manual testing only; it does not train or modify the model.
"""

from datetime import date, datetime
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cycle_length_model.joblib"
METADATA_PATH = ROOT / "models" / "cycle_length_model_metadata.json"

DEFAULT_PAYLOAD = {
    "date_of_birth": "2000-08-15",
    "menarche_age": 13,
    "height_cm": 165.1,
    "weight_kg": 60.0,
    "sleep_hours": 6.0,
    "stress_level": 3,
    "exercise_frequency": 1,
    "uses_medication_or_contraceptive": False,
    "cycle_lengths": [28, 29, 27],
    "period_lengths": [5, 5, 6],
}


def require_number(name, value, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be an integer.")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return int(value) if integer else float(value)


def validate_history(name, values, low, high):
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list ordered oldest-to-newest.")
    return [require_number(f"{name}[{index}]", value, low, high) for index, value in enumerate(values)]


def get_input(prompt, default_value, converter=str):
    entered = input(f"{prompt} [{default_value}]: ").strip()
    if not entered:
        return default_value
    try:
        return converter(entered)
    except ValueError as error:
        raise ValueError(f"Invalid value for {prompt}: {entered}") from error


def get_history(prompt, default_value):
    default_text = ",".join(map(str, default_value))
    entered = input(f"{prompt} [{default_text}]: ").strip()
    if not entered:
        return default_value
    try:
        return [float(value.strip()) for value in entered.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("Histories must contain comma-separated numeric day values.") from error


def get_boolean(prompt, default_value):
    default_text = "true" if default_value else "false"
    entered = input(f"{prompt} [{default_text}]: ").strip().lower()
    if not entered:
        return default_value
    if entered not in {"true", "false"}:
        raise ValueError("Enter true or false.")
    return entered == "true"


def collect_payload():
    print("\n=== NAVYA Model Prediction Tester ===")
    print("Press Enter to use the value shown in brackets.")
    print("Cycle and period histories must be entered oldest-to-newest.")
    print("Stress: 1=Very Low, 2=Low, 3=Moderate, 4=High, 5=Very High")
    print("Exercise: 0=Never, 1=1-2 days/week, 2=3+ days/week\n")
    return {
        "date_of_birth": get_input("Date of birth (YYYY-MM-DD)", DEFAULT_PAYLOAD["date_of_birth"]),
        "menarche_age": get_input("Menarche age", DEFAULT_PAYLOAD["menarche_age"], int),
        "height_cm": get_input("Height (cm)", DEFAULT_PAYLOAD["height_cm"], float),
        "weight_kg": get_input("Weight (kg)", DEFAULT_PAYLOAD["weight_kg"], float),
        "sleep_hours": get_input("Sleep hours", DEFAULT_PAYLOAD["sleep_hours"], float),
        "stress_level": get_input("Stress level (1-5)", DEFAULT_PAYLOAD["stress_level"], int),
        "exercise_frequency": get_input("Exercise frequency (0-2)", DEFAULT_PAYLOAD["exercise_frequency"], int),
        "uses_medication_or_contraceptive": get_boolean(
            "Uses medication or contraceptive? (true/false)",
            DEFAULT_PAYLOAD["uses_medication_or_contraceptive"],
        ),
        "cycle_lengths": get_history("Cycle lengths (oldest-to-newest)", DEFAULT_PAYLOAD["cycle_lengths"]),
        "period_lengths": get_history("Period lengths (oldest-to-newest)", DEFAULT_PAYLOAD["period_lengths"]),
    }


def build_feature_row(payload, metadata):
    required = set(DEFAULT_PAYLOAD)
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    try:
        dob = datetime.strptime(payload["date_of_birth"], "%Y-%m-%d").date()
    except (TypeError, ValueError) as error:
        raise ValueError("date_of_birth must use YYYY-MM-DD.") from error

    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    age = require_number("age_years", age, 8, 100, integer=True)
    menarche = require_number("menarche_age", payload["menarche_age"], 7, age, integer=True)
    height = require_number("height_cm", payload["height_cm"], 100, 230)
    weight = require_number("weight_kg", payload["weight_kg"], 25, 250)
    sleep = require_number("sleep_hours", payload["sleep_hours"], 1, 24)
    stress = require_number("stress_level", payload["stress_level"], 1, 5, integer=True)
    exercise = require_number("exercise_frequency", payload["exercise_frequency"], 0, 2, integer=True)
    medication = payload["uses_medication_or_contraceptive"]
    if not isinstance(medication, bool):
        raise ValueError("uses_medication_or_contraceptive must be true or false.")

    cycles = validate_history("cycle_lengths", payload["cycle_lengths"], 15, 60)
    if len(cycles) < 3:
        return None, cycles
    recent_cycles = cycles[-3:]

    periods = validate_history("period_lengths", payload["period_lengths"], 1, 14)[-3:]
    period_fill = float(np.mean(periods)) if periods else float(metadata["training_period_history_median"])
    padded_periods = periods + [period_fill] * (3 - len(periods))
    prev_period_1, prev_period_2, prev_period_3 = reversed(padded_periods)
    prev_cycle_1, prev_cycle_2, prev_cycle_3 = reversed(recent_cycles)

    row = {
        "age_years": age,
        "menarche_age": menarche,
        "height_cm": height,
        "weight_kg": weight,
        "bmi": weight / (height / 100) ** 2,
        "sleep_hours": sleep,
        "stress_level": stress,
        "exercise_frequency": exercise,
        "uses_medication_or_contraceptive": int(medication),
        "prev_cycle_1": prev_cycle_1,
        "prev_cycle_2": prev_cycle_2,
        "prev_cycle_3": prev_cycle_3,
        "avg_previous_cycle_length": float(np.mean(recent_cycles)),
        "std_previous_cycle_length": float(np.std(recent_cycles, ddof=0)),
        "prev_period_1": prev_period_1,
        "prev_period_2": prev_period_2,
        "prev_period_3": prev_period_3,
        "avg_previous_period_length": float(np.mean([prev_period_1, prev_period_2, prev_period_3])),
    }
    features = metadata["feature_order"]
    return pd.DataFrame([[row[feature] for feature in features]], columns=features), cycles


def predict(payload, pipeline, metadata):
    features, cycles = build_feature_row(payload, metadata)
    if len(cycles) == 0:
        estimate = float(metadata["training_cycle_history_median"])
        return {"predicted_cycle_length_days": estimate, "prediction_method": "fallback_population_median", "prediction_status": "not_enough_personal_data"}, None
    if len(cycles) < 3:
        estimate = float(np.mean(cycles))
        return {"predicted_cycle_length_days": estimate, "prediction_method": "fallback_history_average", "prediction_status": "not_enough_cycle_history"}, None

    recent_cycles = cycles[-3:]
    lower = float(metadata["supported_cycle_lower_bound"])
    upper = float(metadata["supported_cycle_upper_bound"])
    if min(recent_cycles) < lower or max(recent_cycles) > upper:
        estimate = float(np.mean(recent_cycles))
        return {
            "predicted_cycle_length_days": estimate,
            "prediction_method": "fallback_history_average",
            "prediction_status": "outside_training_distribution",
            "supported_cycle_range_days": [lower, upper],
        }, features

    raw_prediction = float(pipeline.predict(features)[0])
    prediction = float(np.clip(raw_prediction, 15, 60))
    q_hat = float(metadata["q_hat_days"])
    return {
        "raw_prediction": raw_prediction,
        "predicted_cycle_length_days": prediction,
        "rounded_prediction": round(prediction),
        "prediction_method": "machine_learning",
        "prediction_status": "supported",
        "prediction_interval": {
            "lower_days": max(15.0, prediction - q_hat),
            "upper_days": min(60.0, prediction + q_hat),
            "coverage": float(metadata["coverage"]),
        },
    }, features


def print_prediction_card(result, features):
    """Print a presentation-friendly prediction summary without extra packages."""
    width = 68
    print("\n" + "═" * width)
    print("                 NAVYA • CYCLE LENGTH PREDICTION")
    print("═" * width)

    method_labels = {
        "machine_learning": "Final machine-learning model",
        "fallback_history_average": "Personal-history fallback",
        "fallback_population_median": "Population-median fallback",
    }
    status_labels = {
        "supported": "Supported by training data",
        "not_enough_personal_data": "Not enough personal cycle history",
        "not_enough_cycle_history": "Fewer than three completed cycles",
        "outside_training_distribution": "Outside the supported training range",
    }
    prediction = result["predicted_cycle_length_days"]
    print(f"\n  Predicted next cycle length : {prediction:.1f} days")
    print(f"  Rounded app display          : {round(prediction)} days")
    print(f"  Prediction method            : {method_labels[result['prediction_method']]}")
    print(f"  Prediction status            : {status_labels[result['prediction_status']]}")

    interval = result.get("prediction_interval")
    if interval:
        print("\n  90% prediction interval")
        print(f"  Expected range               : {interval['lower_days']:.1f} – {interval['upper_days']:.1f} days")
        print("  Meaning                      : A realistic range, not a guaranteed date.")

    supported_range = result.get("supported_cycle_range_days")
    if supported_range:
        print("\n  Note")
        print(f"  Model support range          : {supported_range[0]:.1f} – {supported_range[1]:.1f} days")
        print("  The result uses the user's own recent-cycle average rather than an unreliable ML estimate.")

    print("\n" + "─" * width)
    if features is not None:
        print("  Feature row constructed successfully: 18 features in the exported model order.")
    else:
        print("  No ML feature row was created because the fallback policy was used.")
    print("═" * width)


def main():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Run Notebook 10 first: final model or metadata is missing.")

    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    actual_estimator = pipeline.named_steps["model"].__class__.__name__
    if actual_estimator != metadata["selected_estimator"]:
        raise RuntimeError("Model and metadata do not match. Re-run Notebook 10 before testing.")

    print(f"Loaded final model: {actual_estimator}")
    payload = collect_payload()
    result, features = predict(payload, pipeline, metadata)

    print_prediction_card(result, features)
    if features is not None:
        print("\nExact 18-feature row supplied to the exported model:")
        print(features.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"\nPrediction error: {error}")
