import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force a non-interactive backend
import matplotlib.pyplot as plt
import joblib
from shiny import App, render, reactive, ui
from patsy import dmatrix

# --- 1. LOAD PRE-TRAINED ARTIFACTS ---
try:
    rsf = joblib.load("model.joblib")
    imputer = joblib.load("imputer.joblib")
    scaler = joblib.load("scaler.joblib")
    calibrator = joblib.load("calibrator.joblib") 
    data_loaded = True
    error_log = "System ready."
except Exception as e:
    data_loaded = False
    error_log = f"Error loading model files: {e}"

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
                    ui.markdown("**⚠️ Medical Disclaimer:** This tool is for educational purposes only."),
                    style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; font-size: 0.9em; border-left: 5px solid #6c757d;"
                )
            ),
            ui.nav_panel("Health Guidance", ui.output_ui("rec_list")),
            ui.nav_panel("Patient Data", 
                ui.div(style="margin-top: 20px;"),
                ui.output_text_verbatim("formatted_inputs") 
            ),
            ui.nav_panel("System Logs", ui.output_ui("log_display"))
        )
    )
)

# --- 3. SERVER LOGIC ---
def server(input, output, session):

    @reactive.calc
    @reactive.event(input.run) # This is the "gatekeeper"
    def process_data():
        if not data_loaded: return None

        # Map UI inputs to the 12 raw features used in Period 1
        feature_map = {
            "AGE": "age", "SEX": "sex", "SYSBP": "sysbp", "DIABP": "diabp",
            "TOTCHOL": "chol", "BMI": "bmi", "GLUCOSE": "glucose", 
            "CURSMOKE": "smoker", "CIGPDAY": "cigs", "DIABETES": "diabetes",
            "BPMEDS": "bpmeds", "PREVHYP": "prevhyp"
        }
        
        user_vals = {}
        imputed_list = []
        for feat, inp_id in feature_map.items():
            val = getattr(input, inp_id)()
            if val is None or val == "":
                user_vals[feat] = np.nan
                imputed_list.append(feat)
            else:
                user_vals[feat] = float(val)
            
        raw_df = pd.DataFrame([user_vals])

        # Impute raw values (Scale -> Impute -> Inverse Scale)
        # This ensures logs and MAP are calculated on "clean" data
        X_scaled = scaler.transform(raw_df) 
        X_imputed = imputer.transform(X_scaled)
        # Convert back to real numbers so math (Logs/MAP) works
        clean_df = pd.DataFrame(scaler.inverse_transform(X_imputed), columns=raw_df.columns)

        # Create Engineered Features (Must match training logic)
        clean_df["LOG_TOTCHOL"] = np.log1p(clean_df["TOTCHOL"])
        clean_df["LOG_CIGS"] = np.log1p(clean_df["CIGPDAY"])
        clean_df["MAP"] = clean_df["DIABP"] + (1/3) * (clean_df["SYSBP"] - clean_df["DIABP"])

        # Generate Age Splines
        age_splines = dmatrix("bs(clean_df.AGE, df=4, include_intercept=False)", 
                            data=clean_df, return_type="dataframe")
        age_splines.columns = [f"AGE_SPLINE_{i}" for i in range(age_splines.shape[1])]
        
        # Assemble final vector in the EXACT order of the training 'model_cols'
        final_features = [
            "SEX", "MAP", "BMI", "LOG_TOTCHOL", "GLUCOSE", 
            "CURSMOKE", "LOG_CIGS", "DIABETES", "BPMEDS", "PREVHYP"
        ] + list(age_splines.columns)
        
        X_input = pd.concat([clean_df, age_splines], axis=1)[final_features]

        # Predict and Calibrate
        raw_risk_score = rsf.predict(X_input)
        calibrated_prob = calibrator.predict_proba(raw_risk_score.reshape(-1, 1))[0][1]
        surv_funcs = rsf.predict_survival_function(X_input, return_array=False)
        
        impacts = []
        improvements = {
            "SYSBP": {"target": 120.0, "label": "Lowering Blood Pressure to 120"},
            "TOTCHOL": {"target": 180.0, "label": "Lowering Cholesterol to 180"},
            "CURSMOKE": {"target": 0.0, "label": "Quitting Smoking"},
            "BMI": {"target": 22.0, "label": "Reaching a healthy BMI (22)"}
        }

        base_risk_pct = calibrated_prob * 100

        for feat, info in improvements.items():
            if clean_df[feat].values[0] > info["target"]:
                sim_df = clean_df.copy()
                sim_df[feat] = info["target"]
                
                # RE-CALCULATE dependent features for simulation
                sim_df["LOG_TOTCHOL"] = np.log1p(sim_df["TOTCHOL"])
                sim_df["LOG_CIGS"] = np.log1p(sim_df["CIGPDAY"])
                sim_df["MAP"] = sim_df["DIABP"] + (1/3) * (sim_df["SYSBP"] - sim_df["DIABP"])
                
                # Assemble simulation vector (Keep splines from baseline)
                X_sim = pd.concat([sim_df, age_splines], axis=1)[final_features]
                sim_risk = calibrator.predict_proba(rsf.predict(X_sim).reshape(-1, 1))[0][1] * 100
                
                reduction = base_risk_pct - sim_risk
                if reduction > 0.5:
                    impacts.append({"label": info["label"], "reduction": reduction})

        return {
            "rsf": surv_funcs[0], 
            "calibrated_risk": calibrated_prob * 100, 
            "imputed": imputed_list,
            "user_vals": user_vals,
            "guidance": sorted(impacts, key=lambda x: x["reduction"], reverse=True) 
        }

    @render.ui
    def risk_badge():
        res = process_data()
        if res is None: return ui.p("Adjust factors and click 'Analyze Risk Factors'.")
        
        risk_pct = res["calibrated_risk"]
        color = "#e74c3c" if risk_pct > 20 else "#f1c40f" if risk_pct > 10 else "#27ae60"
        
        return ui.div(
            ui.h3(f"Predicted 10-Year Risk: {risk_pct:.1f}%"),
            style=f"padding: 20px; color: white; background-color: {color}; border-radius: 12px; text-align: center;"
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
        return fig

    @render.text
    def formatted_inputs():
        res = process_data()
        if not res: 
            return "No data analyzed yet. Please click 'Analyze Risk Factors' first."
        
        vals = res["user_vals"]
        
        # Define how to translate numbers back to labels
        label_map = {
            "SEX": {1.0: "M", 2.0: "F"},
            "CURSMOKE": {1.0: "Yes", 0.0: "No"},
            "DIABETES": {1.0: "Yes", 0.0: "No"},
            "BPMEDS": {1.0: "Yes", 0.0: "No"},
            "PREVHYP": {1.0: "Yes", 0.0: "No"}
        }
        
        lines = []
        for key, val in vals.items():
            if pd.isna(val):
                continue
                
            # Check if the column needs a label translation
            if key in label_map:
                display_val = label_map[key].get(val, val)
            # Clean up floats that should be integers (like Age or Cigs/Day)
            elif val == int(val):
                display_val = int(val)
            else:
                display_val = val
                
            lines.append(f"{key:<10}: {display_val}")
        
        return "--- PATIENT DATA SUMMARY ---\n" + "\n".join(lines)


    @render.ui
    def log_display():
        return ui.markdown(f"**Log Status:** `{error_log}`")

    @render.ui
    def imputation_alert():
        res = process_data()
        if res and res["imputed"]:
            cols = ", ".join(res["imputed"])
            return ui.div(f"ℹ️ Note: Missing values for ({cols}) were estimated using KNN. Fill in all known fields for more accurate results.", 
                          style="color: #856404; background-color: #fff3cd; padding: 12px; border-radius: 6px; margin-top: 10px;")

    @render.ui
    def rec_list():
        res = process_data()
        if res is None:
            return ui.p("Please click 'Analyze Risk Factors' to see guidance.")
        
        impacts = res["guidance"]
        
        if not impacts:
            return ui.div(
                ui.p("Your vitals are within healthy ranges, or you haven't run the analysis yet."),
                ui.p("Continue maintaining a healthy lifestyle and regular checkups.")
            )
            
        # Create a list of recommendations based on model impact
        list_items = []
        for item in impacts:
            list_items.append(
                ui.tags.li(
                    ui.markdown(f"**{item['label']}**: Could reduce your 10-year risk by **{item['reduction']:.1f}%**")
                )
            )
            
        return ui.div(
            ui.h4("Your Top Improvement Opportunities:"),
            ui.p("Based on the model, these changes would have the biggest impact on your heart health:"),
            ui.tags.ul(list_items, style="font-size: 1.1em; line-height: 1.6;"),
            ui.hr(),
            ui.p("💡 *These calculations are simulated by the AI model based on your unique profile.*", style="font-size: 0.8em; color: #666;")
        )


app = App(app_ui, server)
