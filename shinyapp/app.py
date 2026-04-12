import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shiny import App, render, reactive, ui
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# --- 1. DATA & ML INITIALIZATION ---
data_loaded = False
error_log = ""

# Model, Scaler, and Imputer Setup
rsf = RandomSurvivalForest(
    n_estimators=500,
    min_samples_split=10,
    min_samples_leaf=15,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)
scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=5)

try:
    # Processing the local framingham_data.csv
    df = pd.read_csv("framingham_data.csv")
    df.columns = [col.upper() for col in df.columns]

    # Required clinical columns for both features and targets
    required_cols = [
        "AGE", "SEX", "SYSBP", "DIABP", "TOTCHOL", "BMI",
        "GLUCOSE", "CURSMOKE", "CIGPDAY", "DIABETES", "BPMEDS", "PREVHYP",
        "TIMECVD", "CVD"
    ]
    
    if all(c in df.columns for c in required_cols):
        df = df.dropna(subset=required_cols)
        features = required_cols[:-2] 
        
        X = df[features].astype(float)
        y = Surv.from_arrays(
            event=df["CVD"].astype(bool),
            time=df["TIMECVD"].astype(float)
        )

        # 80/20 Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit Pipeline
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        imputer.fit(X_train_scaled)
        rsf.fit(X_train_scaled, y_train)
        data_loaded = True
    else:
        error_log = f"Missing: {[c for c in required_cols if c not in df.columns]}"
except Exception as e:
    error_log = str(e)

# UI Helper
def input_row(id, label, value, is_select=False, choices=None):
    widget = ui.input_select(id, None, choices) if is_select else ui.input_numeric(id, None, value)
    return ui.row(
        ui.column(7, ui.span(label), widget),
        ui.column(5, ui.input_checkbox(f"has_{id}", "Known", value=True)),
        style="margin-bottom: 5px; align-items: center;"
    )

# --- 2. USER INTERFACE ---
app_ui = ui.page_fluid(
    ui.panel_title("Heart Disease Risks and Recommendations"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_action_button("run", "Analyze Risk Factors", class_="btn-primary w-100"),
            ui.hr(),
            ui.div(
                ui.h4("Patient Vitals"),
                ui.p("Uncheck if factor is unknown:", style="font-size: 0.85em; color: #666;"),
                input_row("age", "Age", 45),
                input_row("sex", "Sex", None, True, {"1": "M", "2": "F"}),
                input_row("sysbp", "Sys BP", 120),
                input_row("diabp", "Dia BP", 80),
                input_row("chol", "Tot Chol", 210),
                input_row("bmi", "BMI", 25.0),
                input_row("glucose", "Glucose", 80),
                input_row("cigs", "Cigs/Day", 0),
                input_row("smoker", "Smoker", None, True, {"0": "No", "1": "Yes"}),
                input_row("diabetes", "Diabetes", None, True, {"0": "No", "1": "Yes"}),
                input_row("bpmeds", "On BP Meds", None, True, {"0": "No", "1": "Yes"}),
                input_row("prevhyp", "Prevalent Hyp", None, True, {"0": "No", "1": "Yes"}),
                style="max-height: 70vh; overflow-y: auto; padding-right: 10px;"
            ),
            width=380
        ),
        ui.navset_tab(
            ui.nav_panel("Risk Forecast",
                ui.div(style="margin-top: 20px;"),
                ui.output_ui("risk_badge"),
                ui.output_plot("survival_plot"),
                ui.output_ui("imputation_alert"),
                ui.hr(),
                ui.div(
                    ui.markdown("**⚠️ Medical Disclaimer:** This tool is for educational purposes only. Always consult a healthcare professional."),
                    style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; font-size: 0.9em; border-left: 5px solid #6c757d;"
                )
            ),
            ui.nav_panel("Health Guidance", ui.output_ui("rec_list")),
            ui.nav_panel("System Logs", ui.markdown(f"**Log:** `{error_log}`"))
        )
    )
)

# --- 3. SERVER LOGIC ---
def server(input, output, session):

    @reactive.calc
    @reactive.event(input.run)
    def process_data():
        if not data_loaded: return None
        
        feature_map = {
            "AGE": "age", "SEX": "sex", "SYSBP": "sysbp", "DIABP": "diabp",
            "TOTCHOL": "chol", "BMI": "bmi", "GLUCOSE": "glucose", 
            "CURSMOKE": "smoker", "CIGPDAY": "cigs", "DIABETES": "diabetes",
            "BPMEDS": "bpmeds", "PREVHYP": "prevhyp"
        }
        
        user_vals = {}
        imputed_list = []
        for feat, inp_id in feature_map.items():
            if getattr(input, f"has_{inp_id}")():
                user_vals[feat] = float(getattr(input, inp_id)())
            else:
                user_vals[feat] = np.nan
                imputed_list.append(feat)
        
        user_df = pd.DataFrame([user_vals])
        scaled_user = scaler.transform(user_df)
        imputed_user = imputer.transform(scaled_user)
        
        # FIX: return_array=False ensures we get StepFunction objects with .x and .y
        surv_funcs = rsf.predict_survival_function(imputed_user, return_array=False)
        return {"rsf": surv_funcs[0], "imputed": imputed_list}

    @render.ui
    def risk_badge():
        res = process_data()
        if res is None: return ui.p("Adjust factors and click 'Analyze Risk Factors'.")
        
        # Use .x and .y from the StepFunction object
        rsf_func = res["rsf"]
        idx = np.abs(rsf_func.x - 3652).argmin()
        risk_pct = (1 - rsf_func.y[idx]) * 100
        color = "#e74c3c" if risk_pct > 20 else "#f1c40f" if risk_pct > 10 else "#27ae60"
        
        return ui.div(
            ui.h3(f"Predicted 10-Year Risk: {risk_pct:.1f}%"),
            style=f"padding: 20px; color: white; background-color: {color}; border-radius: 12px; text-align: center; margin-bottom: 10px;"
        )

    @render.plot
    def survival_plot():
        res = process_data()
        if res is None: return None
        rsf_func = res["rsf"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.step(rsf_func.x / 365.25, rsf_func.y, where="post", color='#2980b9', lw=2.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Years into Future")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)
        return fig

    @render.ui
    def imputation_alert():
        res = process_data()
        if res and res["imputed"]:
            return ui.div(f"ℹ️ Estimated via KNN: {', '.join(res['imputed'])}", 
                          style="color: #856404; background-color: #fff3cd; padding: 12px; border-radius: 6px;")

    @render.ui
    def rec_list():
        res = process_data()
        if not res: return ui.p("Run analysis to see guidance.")
        recs = ["Maintain regular GP screenings."]
        if input.has_sysbp() and input.sysbp() > 140: recs.append("📉 **Blood Pressure:** Focus on reduction.")
        return ui.tags.ul([ui.tags.li(ui.markdown(r)) for r in recs])

app = App(app_ui, server)
