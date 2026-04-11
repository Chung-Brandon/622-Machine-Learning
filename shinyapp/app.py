from shiny import App, render, ui, reactive
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. MODELING LIBRARIES ---
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest

# --- DATA LOADING & TRAINING ---
data_loaded = False
cph = CoxPHFitter()
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, n_jobs=-1)
error_log = ""
detected_cols = []

try:
    df = pd.read_csv("framingham_data.csv").dropna()
    df.columns = [col.upper() for col in df.columns]
    detected_cols = df.columns.tolist()

    # Column Mapping
    event_col = next((c for c in ["TENYEARCHD", "ANYCHD", "CVD"] if c in df.columns), None)
    duration_col = next((c for c in ["TIME", "PERIOD", "DAYS"] if c in df.columns), None)
    features = ["AGE", "TOTCHOL", "CURSMOKE"]

    if event_col and duration_col and all(f in df.columns for f in features):
        # Fit Cox (Lifelines)
        model_df = df[[duration_col, event_col] + features]
        cph.fit(model_df, duration_col=duration_col, event_col=event_col)

        # Fit Random Forest (scikit-survival)
        y = np.empty(len(model_df), dtype=[('f0', bool), ('f1', float)])
        y['f0'] = model_df[event_col].astype(bool)
        y['f1'] = model_df[duration_col].astype(float)
        rsf.fit(model_df[features], y)
        
        data_loaded = True
except Exception as e:
    error_log = str(e)

# --- 2. USER INTERFACE ---
DEBUG_MODE = True 

app_ui = ui.page_fluid(
    ui.panel_title("Heart Disease Predictor: Cox vs Random Forest"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Patient Vitals"),
            ui.input_numeric("age", "Age", value=45, min=30, max=95),
            ui.input_numeric("chol", "Total Cholesterol", value=210, min=100),
            ui.input_select("smoker", "Smoker Status", {"0": "Non-Smoker", "1": "Current Smoker"}),
            ui.input_checkbox("show_rsf", "Compare with Random Forest", value=True),
            ui.input_action_button("run", "Analyze Survival", class_="btn-primary"),
            ui.hr(),
            ui.markdown(f"**Status:** {'✅ Model Active' if data_loaded else '❌ Error'}")
        ),
        ui.navset_tab(
            ui.nav_panel("Survival Forecast",
                ui.h3("Probability of Remaining Disease-Free"),
                ui.output_plot("survival_plot"),
                ui.output_ui("risk_badge")
            ),
            ui.nav_panel("Recommendations",
                ui.h3("Health Guidance"),
                ui.output_ui("rec_list")
            ),
            *(
                [ui.nav_panel("Data Debugger",
                    ui.h4("CSV Diagnostics"),
                    ui.markdown(f"**Found Columns:** `{detected_cols}`"),
                    ui.markdown(f"**Error Details:** `{error_log if error_log else 'None'}`"),
                    ui.output_table("preview")
                )] if DEBUG_MODE else []
            )
        )
    )
)

# --- 3. SERVER LOGIC ---
def server(input, output, session):

    @reactive.Calc
    @reactive.event(input.run)
    def predict_data():
        if not data_loaded: return None
        # Add new inputs here 
        user_input = pd.DataFrame({
            'AGE': [input.age()],
            'TOTCHOL': [input.chol()],
            'CURSMOKE': [int(input.smoker())]
        })

        # Get Cox Prediction
        cox_curve = cph.predict_survival_function(user_input)
        
        # Get RSF Prediction
        rsf_funcs = rsf.predict_survival_function(user_input)
        # rsf_funcs[0].x is time, rsf_funcs[0].y is survival prob
        
        return {"cox": cox_curve, "rsf": rsf_funcs[0]}

    @output
    @render.plot
    def survival_plot():
        preds = predict_data()
        if preds is None: return None
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Plot Cox (Standard Line)
        cox = preds["cox"]
        years_cox = cox.index / 365.25
        ax.plot(years_cox, cox.iloc[:, 0], color='#2c3e50', linewidth=3, label="Cox Model")
        
        # Plot Random Forest (Step Function)
        if input.show_rsf():
            rsf_func = preds["rsf"]
            years_rsf = rsf_func.x / 365.25
            ax.step(years_rsf, rsf_func.y, where="post", color='#e74c3c', 
                    linewidth=2, linestyle='--', label="Random Forest")
        
        ax.set_xlabel("Years into the Future")
        ax.set_ylabel("Probability of No Heart Disease")
        ax.set_ylim(0.5, 1.01) # Expanded range to see model differences
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    @output
    @render.ui
    def risk_badge():
        preds = predict_data()
        if preds is None: return None
        
        # Use Cox model for the primary badge (10-year / 3652 days)
        curve = preds["cox"]
        idx = np.abs(curve.index - 3652).argmin()
        risk_pct = (1 - curve.iloc[idx, 0]) * 100
        
        color = "#e74c3c" if risk_pct > 20 else "#f1c40f" if risk_pct > 10 else "#27ae60"
        
        return ui.div(
            ui.h4(f"Your 10-Year Heart Disease Risk: {risk_pct:.1f}%"),
            ui.p("(Based on Cox Proportional Hazards Model)"),
            style=f"padding: 20px; color: white; background-color: {color}; border-radius: 8px; text-align: center;"
        )

    @output
    @render.ui
    def rec_list():
        # Add recommendations here 
        preds = predict_data()
        if preds is None: return ui.p("Predict to see results.")
        # Using Cox Model coefficients to show impact
        coefs = cph.params_
        recs = [f"Baseline monitoring for age {input.age()}."]
        if input.chol() > 240:
            recs.append("📉 **Cholesterol:** High levels. Consider dietary changes.")
        if input.smoker() == "1":
            impact = np.exp(coefs['CURSMOKE']) # The Hazard Ratio
            recs.append(f"🚭 **Smoking:** Quitting could reduce your risk by {((impact-1)*100):.0f}%")
        
        return ui.tags.ul([ui.tags.li(ui.markdown(r)) for r in recs])

    @output
    @render.table
    def preview():
        return df.head() if data_loaded else None

app = App(app_ui, server)
