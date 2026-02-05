from utils import db_connect
engine = db_connect()

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CSV_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=809&path=housing.csv"
CSV_PATH = DATA_DIR / "housing.csv"

KMEANS_PATH = MODELS_DIR / "housing_kmeans.joblib"
SCALER_PATH = MODELS_DIR / "housing_scaler.joblib"
RF_PATH = MODELS_DIR / "housing_cluster_classifier.joblib"

COLS = ["Latitude", "Longitude", "MedInc"]

def load_dataset():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.read_csv(CSV_URL)
        df.to_csv(CSV_PATH, index=False)

    df = df[COLS].copy()

    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()

    return df

def train_models(df):
    X = df[COLS]

    X_train_raw, X_test_raw = train_test_split(
        X,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    kmeans = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_train)

    y_train = kmeans.predict(X_train)
    y_test = kmeans.predict(X_test)

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train_raw, y_train)

    pred = rf.predict(X_test_raw)
    acc = float(accuracy_score(y_test, pred))

    return scaler, kmeans, rf, acc

def predict_cluster(scaler, kmeans, rf, values):
    row = pd.DataFrame([values], columns=COLS)
    km_cluster = int(kmeans.predict(scaler.transform(row))[0])
    rf_cluster = int(rf.predict(row)[0])
    return {"kmeans_cluster": km_cluster, "rf_cluster": rf_cluster}

def main():
    df = load_dataset()
    scaler, kmeans, rf, acc = train_models(df)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(kmeans, KMEANS_PATH)
    joblib.dump(rf, RF_PATH)

    print("CSV:", CSV_PATH)
    print("Scaler:", SCALER_PATH)
    print("KMeans:", KMEANS_PATH)
    print("RF:", RF_PATH)
    print("RF accuracy vs KMeans-labels:", acc)

    demo = predict_cluster(
        scaler,
        kmeans,
        rf,
        [34.05, -118.25, 4.5]
    )
    print("Demo prediction:", demo)


if __name__ == "__main__":
    main()
