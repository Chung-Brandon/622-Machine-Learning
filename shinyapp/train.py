"""
This file is for building model artifacts.
Artifacts reduce the load on the live environment.
Run this whenever a change is made to a file to generate .joblib files.
Upload .loblib files into the shinyapp folder.
"""
import pandas as pd
import numpy as np
import joblib
from patsy import dmatrix

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.linear_model import LogisticRegression # For Platt Scaling
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def train_and_save_model():
    try:
        # Load Data
        df = pd.read_csv("framingham_data.csv")

        # If the same patient appears in multiple periods, that 
        # violates the assumption of independence
        df = df[df["PERIOD"] == 1]
        df.columns = [col.upper() for col in df.columns]

        raw_features = [
            "AGE", "SEX", "SYSBP", "DIABP", "TOTCHOL", 
            "BMI", "GLUCOSE", "CURSMOKE", "CIGPDAY", "DIABETES", "BPMEDS", "PREVHYP"
        ]
        targets = ["TIMECVD", "CVD"]

        # Clean and Prep
        df = df.dropna(subset=targets)
        X_raw = df[raw_features]
        y = Surv.from_arrays(
            event=df["CVD"].astype(bool),
            time=df["TIMECVD"].astype(float)
        )

        # Train-Test Split (Standard 80/20)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

        # Fit Scaler and Imputer on raw training data
        scaler = StandardScaler()
        imputer = KNNImputer(n_neighbors=5)
        
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_train_imputed = imputer.fit_transform(X_train_scaled)

        def engineer_features(imputed_array, original_cols):
            # Convert back to real numbers for Logs and MAP
            temp_df = pd.DataFrame(scaler.inverse_transform(imputed_array), columns=original_cols)
            
            # Log Transformations to reduce the impact of outliers
            temp_df["LOG_TOTCHOL"] = np.log1p(temp_df["TOTCHOL"])
            temp_df["LOG_CIGS"] = np.log1p(temp_df["CIGPDAY"])
            # Formula for Mean Arterial Pressure found at: https://www.ncbi.nlm.nih.gov/books/NBK538226/
            # MAP = DP + 1/3(SP – DP) or MAP = DP + 1/3(PP)
            # Feature created to reduce feature redundancy and multicollinearity concerns
            temp_df["MAP"] = temp_df["DIABP"] + (1/3) * (temp_df["SYSBP"] - temp_df["DIABP"])
            
            # Age Splines to reduce jaggedness
            age_splines = dmatrix("bs(temp_df.AGE, df=4, include_intercept=False)", 
                                  data=temp_df, return_type="dataframe")
            age_splines.columns = [f"AGE_SPLINE_{i}" for i in range(age_splines.shape[1])]
            
            # Final Model Column Set
            model_cols = [
                "SEX", "MAP", "BMI", "LOG_TOTCHOL",
                "GLUCOSE", "CURSMOKE", "LOG_CIGS", "DIABETES", "BPMEDS", "PREVHYP"
            ] + list(age_splines.columns)
            
            return pd.concat([temp_df, age_splines], axis=1)[model_cols]
        
        X_train_final = engineer_features(X_train_imputed, raw_features)

        # Train the Random Survival Forest
        # We use n_jobs=-1 here to speed up local training
        print("Training Random Survival Forest")
        rsf = RandomSurvivalForest(
            n_estimators=100, # Reduce size of model to host app
            max_depth=8, # Limits the size of each tree
            min_samples_split=20,
            min_samples_leaf=15,
            n_jobs=-1,
            random_state=42
        )
        rsf.fit(X_train_final, y_train)

        print("Calibrating model (Platt Scaling)")
        # Get raw risk scores for the test set
        # (Higher score = higher risk of the event occurring)
        X_test_imputed = imputer.transform(scaler.transform(X_test_raw))
        X_test_final = engineer_features(X_test_imputed, raw_features)
        
        raw_scores_test = rsf.predict(X_test_final)
        event_within_10y = (y_test["event"]) & (y_test["time"] <= 3650)

        # Fit Logistic Regression
        calibrator = LogisticRegression()
        calibrator.fit(raw_scores_test.reshape(-1, 1), event_within_10y.astype(int))

        # Save Artifacts for the Shiny App
        print("Saving artifacts")
        joblib.dump(rsf, "model.joblib", compress=9)
        joblib.dump(imputer, "imputer.joblib")
        joblib.dump(scaler, "scaler.joblib")
        joblib.dump(calibrator, "calibrator.joblib")

        print("Artifacts created: model.joblib, imputer.joblib, scaler.joblib, calibrator.joblib")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    train_and_save_model()
