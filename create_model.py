import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():

    # Vijayawada Landmarks
    landmarks = [
        "Benz Circle", "MG Road", "Labbipet", "Governorpet",
        "Auto Nagar", "Vidhyadharpuram", "Gollapudi",
        "Enikepadu", "Rama Krishna Puram", "Hanumanpet"
    ]

    np.random.seed(42)
    n_samples = 2000  # MORE DATA → BETTER MODEL

    # Synthetic dataset
    data = {
        "source_idx": np.random.randint(0, len(landmarks), n_samples),
        "destination_idx": np.random.randint(0, len(landmarks), n_samples),
        "time_of_day": np.random.randint(0, 4, n_samples),
        "distance": np.random.uniform(1, 15, n_samples),
        "population_density": np.random.uniform(2000, 8000, n_samples),
        "streetlight_density": np.random.uniform(10, 100, n_samples),
        "police_stations_nearby": np.random.randint(0, 5, n_samples)
    }

    df = pd.DataFrame(data)

    # Generate synthetic safety score (more realistic)
    safety_score = (
        df["streetlight_density"] * 0.3 +
        df["police_stations_nearby"] * 8 +
        (3 - df["time_of_day"]) * 5 +
        (15 - df["distance"]) * 2 +
        np.random.normal(0, 5, n_samples)
    )

    df["safety_score"] = np.clip(safety_score, 0, 100)

    df["safety_label"] = pd.cut(
        df["safety_score"],
        bins=[0, 30, 70, 100],
        labels=["Unsafe", "Moderate", "Safe"]
    )

    df = df.dropna()

    # FEATURES & LABELS
    X = df[["source_idx", "destination_idx", "time_of_day", "distance",
            "population_density", "streetlight_density", "police_stations_nearby"]]
    y = df["safety_label"]

    # TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # MODEL
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    # EVALUATION
    y_pred = model.predict(X_test)

    print("\n--- MODEL RESULTS ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, "safety_model.pkl")
    joblib.dump(landmarks, "landmarks.pkl")

    print("\nModel saved successfully!")

if __name__ == "__main__":
    train_model()
