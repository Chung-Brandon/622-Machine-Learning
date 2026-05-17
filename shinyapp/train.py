"""
This file is for building model artifacts.
Artifacts reduce the load on the live environment.
Run this whenever a change is made to generate new .joblib files.
Upload .joblib files into the shinyapp folder.
The full command assuming Python is on PATH and terminal is in the same directory as this file:
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt && python train.py
Once inside the virtual environment, only the regular Python commands need to be run.
"""
import pandas as pd
import numpy as np
import joblib
from patsy import dmatrix

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from lifelines import CoxPHFitter # Cox Proportional Hazards to recalibrate RSF
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def train_and_save_model():
    try:
        # Load Data
        df = pd.read_csv("framingham_data.csv")

        # If the same patient appears in multiple periods, that violates the assumption of independence
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

        # Fit Scaler on raw training data
        scaler = StandardScaler()
        # Impute by median for most fields, but mode for binary values (BPMEDS)
        mode_cols = ["BPMEDS"]
        median_cols = [c for c in raw_features if c not in mode_cols]
        
        # Get indices for positional imputation (required for ColumnTransformer on arrays)
        bpmeds_idx = [raw_features.index("BPMEDS")]
        median_indices = [raw_features.index(c) for c in median_cols]

        imputer = ColumnTransformer(
            transformers=[
                ("median_imp", SimpleImputer(strategy="median"), median_indices),
                ("mode_imp", SimpleImputer(strategy="most_frequent"), bpmeds_idx),
            ],
            remainder="passthrough"
        )
        
        # Scale and Impute
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_train_imputed_array = imputer.fit_transform(X_train_scaled)
        X_train_imputed_df = pd.DataFrame(X_train_imputed_array, columns=median_cols + mode_cols)

        X_train_imputed = X_train_imputed_df[raw_features].values 
        
        def engineer_features(imputed_array, original_cols):
            # Convert back to real numbers for Logs and MAP
            temp_df = pd.DataFrame(scaler.inverse_transform(imputed_array), columns=original_cols)
            temp_df = temp_df.reset_index(drop=True)
            
            # Log Transformations to reduce the impact of outliers
            temp_df["LOG_TOTCHOL"] = np.log1p(temp_df["TOTCHOL"])
            temp_df["LOG_CIGS"] = np.log1p(temp_df["CIGPDAY"])
            # Formula for Mean Arterial Pressure found at: https://www.ncbi.nlm.nih.gov/books/NBK538226/
            # MAP = DP + 1/3(SP – DP) or MAP = DP + 1/3(PP)
            # Feature created to reduce feature redundancy and multicollinearity concerns
            temp_df["MAP"] = temp_df["DIABP"] + (1/3) * (temp_df["SYSBP"] - temp_df["DIABP"])
            
            # Age Splines to reduce jaggedness
            age_splines = dmatrix("bs(temp_df.AGE, df=4, lower_bound=30, upper_bound=70, include_intercept=False)", # adding bounds to train file for mathematical consistency with the Shiny App
                                  data=temp_df, return_type="dataframe")
            age_splines.columns = [f"AGE_SPLINE_{i}" for i in range(age_splines.shape[1])]
            age_splines = age_splines.reset_index(drop=True)
            
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
            max_depth=15, # Limit the size of each tree
            min_samples_split=4,
            min_samples_leaf=10,
            n_jobs=-1,
            oob_score=True,
            random_state=42
        )
        rsf.fit(X_train_final, y_train)

        print("Calibrating model (Platt Scaling)")
        # Get raw risk scores for the test set
        # (Higher score = higher risk of the event occurring)
        X_test_scaled = scaler.transform(X_test_raw)
        X_test_imputed_array = imputer.transform(X_test_scaled)

        # Re-align test columns just like training
        X_test_imputed_df = pd.DataFrame(X_test_imputed_array, columns=median_cols + mode_cols)
        X_test_imputed_final = X_test_imputed_df[raw_features].values

        # Now engineer features for the final test prediction
        X_test_final = engineer_features(X_test_imputed_final, raw_features)
        
        # Use Out-of-Bag scores from the training set to fit the calibrator without data leakage
        oob_scores_train = rsf.oob_prediction_
        
        # Create a calibration dataset matching the lifelines CoxPHFitter requirement
        calibration_df = pd.DataFrame({
            "time": y_train["time"],  
            "event": y_train["event"].astype(int),
            "rsf_score": oob_scores_train
        })

        # Fit a Cox Proportional Hazards model as the scaler, helps with areas of sparse data
        calibrator = CoxPHFitter()
        calibrator.fit(calibration_df, duration_col="time", event_col="event", formula="bs(rsf_score, df=3, include_intercept=False)")

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
