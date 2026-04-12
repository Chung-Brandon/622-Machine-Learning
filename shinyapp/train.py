"""
This file is for building model artifacts.
Artifacts reduce the load on the live environment.
Run this whenever a change is made to a file to generate .joblib files.
Upload .loblib files into the shinyapp folder.
"""
import pandas as pd
import numpy as np
import joblib
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.linear_model import LogisticRegression # For Platt Scaling
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def train_and_save_model():
    try:
        # Load Data
        print("Loading data...")
        df = pd.read_csv("framingham_data.csv")
        df.columns = [col.upper() for col in df.columns]

        # Define Features and Targets
        features = [
            "AGE", "SEX", "SYSBP", "DIABP", "TOTCHOL", "BMI",
            "GLUCOSE", "CURSMOKE", "CIGPDAY", "DIABETES", "BPMEDS", "PREVHYP"
        ]
        targets = ["TIMECVD", "CVD"]
        required_cols = features + targets

        # Clean and Prep
        df = df.dropna(subset=required_cols)
        X = df[features].astype(float)
        y = Surv.from_arrays(
            event=df["CVD"].astype(bool),
            time=df["TIMECVD"].astype(float)
        )

        # Train-Test Split (Standard 80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit Scaler on raw training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit Imputer on SCALED training data 
        imputer = KNNImputer(n_neighbors=5)
        X_train_imputed = imputer.fit_transform(X_train_scaled)

        # Train the Random Survival Forest
        # We use n_jobs=-1 here to speed up local training
        print("Training Random Survival Forest (this may take a minute)...")
        rsf = RandomSurvivalForest(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )
        rsf.fit(X_train_imputed, y_train)

        print("Calibrating model (Platt Scaling)...")
        # Get raw risk scores for the test set
        # (Higher score = higher risk of the event occurring)
        raw_scores_test = rsf.predict(imputer.transform(scaler.transform(X_test)))
        event_observed_test = y_test["event"].astype(int)

        # Fit Logistic Regression: Raw Score -> Binary Outcome
        calibrator = LogisticRegression()
        calibrator.fit(raw_scores_test.reshape(-1, 1), event_observed_test)

        # Save Artifacts for the Shiny App
        print("Saving artifacts...")
        joblib.dump(rsf, 'model.joblib', compress=9)
        joblib.dump(imputer, 'imputer.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(calibrator, 'calibrator.joblib')

        print("Success! Artifacts created: model.joblib, imputer.joblib, scaler.joblib, calibrator.joblib")

    except FileNotFoundError:
        print("Error: framingham_data.csv not found. Ensure it is in this folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    train_and_save_model()
