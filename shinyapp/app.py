"""
Run this file AFTER train.py or if you have all .joblib artifacts already.
Assuming existing artifacts and a clean environment, run this command from the same directory:
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt && shiny run --reload --launch-browser app.py
Once inside the virtual environment, only the regular Python commands need to be run.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.transforms as mtransforms
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import joblib
from shiny import App, render, reactive, ui, req
from patsy import dmatrix

raw_features = [
    "AGE", "SEX", "SYSBP", "DIABP", "TOTCHOL", 
    "BMI", "GLUCOSE", "CURSMOKE", "CIGPDAY", "DIABETES", "BPMEDS", "PREVHYP"
]

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

# Updated UI Helper to cleanly accept numeric boundaries
def input_row(id, label, value, is_select=False, choices=None, **kwargs):
    widget = ui.input_select(id, None, choices) if is_select else ui.input_numeric(id, None, value, **kwargs)
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
                input_row("age", "Age", 45, min=30, max=69), # adding user input bounds to match training data and prevent spline extrapolation issues
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

                # One version with the plots stacked vertically
                # ui.output_plot("isotype_plot"),
                # ui.output_plot("survival_plot"),

                # ----- Side by side plots -----
                ui.hr(),
                ui.row(
                    ui.column(6, 
                        ui.h5("Visual Risk Distribution Profile", style="text-align: left; color: #34495e; font-weight: bold;"),
                        ui.output_plot("isotype_plot")
                    ),
                    ui.column(6, 
                        ui.h5("10-Year Risk Trend", style="text-align: left; color: #34495e; font-weight: bold;"),
                        ui.output_plot("survival_plot")
                    )
                ),
                # ------ End of side by side code snippet -----

                ui.output_ui("imputation_alert"),
                ui.hr(),
                ui.div(
                    ui.markdown("**⚠️ Medical Disclaimer:** This tool is for educational purposes only."),
                    style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; font-size: 0.9em; border-left: 5px solid #6c757d;"
                ),
                ui.div(
                    ui.markdown(
                        "** Note:** Model trained on adults aged 30–70. Values outside this range are automatically adjusted to the nearest supported age."),
                    style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; font-size: 0.9em; border-left: 5px solid #6c757d;"
                ),
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
    @reactive.event(input.run) 
    def process_data():
        try:
            if not data_loaded: return None
            # Hardcoded to match what was trained originally (10 years)
            horizon_years = 10.0
            target_days = horizon_years * 365.25
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
                # Read the numerical text input value directly
                text_box_value = getattr(input, inp_id)()
                
                # Checks by prefix has_. Adjust this if naming changes are made.
                checkbox_id = f"has_{inp_id}"
                checkbox_active = getattr(input, checkbox_id)() 
                
                # If the checkbox is unchecked (False), force an explicit imputation state.
                if checkbox_active is False or text_box_value is None or text_box_value == "":
                    user_vals[feat] = np.nan
                    imputed_list.append(feat) # Tracks which values were dropped for imputation
                else:
                    user_vals[feat] = float(text_box_value)
                
            raw_df = pd.DataFrame([user_vals])
            scaled_raw = scaler.transform(raw_df) 
            imputed_raw_array = imputer.transform(scaled_raw)

            # Define the lists exactly as they were in train.py
            mode_cols = ["BPMEDS"]
            median_cols = [c for c in raw_features if c not in mode_cols]
            
            # Re-attach names to the scrambled array, then re-sort to raw_features order
            temp_df = pd.DataFrame(imputed_raw_array, columns=median_cols + mode_cols)
            imputed_raw_correct_order = temp_df[raw_features].values

            # Since order is corrected, inverse_transform will map the right values
            clean_vals = scaler.inverse_transform(imputed_raw_correct_order)
            clean_df = pd.DataFrame(clean_vals, columns=raw_features)

            # Create Engineered Features like in training
            clean_df["LOG_TOTCHOL"] = np.log1p(clean_df["TOTCHOL"])
            clean_df["LOG_CIGS"] = np.log1p(clean_df["CIGPDAY"])
            clean_df["MAP"] = clean_df["DIABP"] + (1/3) * (clean_df["SYSBP"] - clean_df["DIABP"])

            # Clamp age to the spline domain used during training
            clean_df["AGE"] = clean_df["AGE"].clip(lower=30.0, upper=69.9)
            
            # Generate Age Splines
            spline_formula = "bs(clean_df.AGE, df=4, lower_bound=30, upper_bound=70, include_intercept=False)"
            age_splines = dmatrix(spline_formula, data=clean_df, return_type="dataframe")
                                
            # Append [1] to shape to target the integer column count (4)
            age_splines.columns = [f"AGE_SPLINE_{i}" for i in range(age_splines.shape[1])]

            # Reset indices to prevent concat mismatch errors
            clean_df = clean_df.reset_index(drop=True)
            age_splines = age_splines.reset_index(drop=True)

            
            # Assemble final vector in the same order of the training "model_cols" or it will not run
            final_features = [
                "SEX", "MAP", "BMI", "LOG_TOTCHOL", "GLUCOSE", 
                "CURSMOKE", "LOG_CIGS", "DIABETES", "BPMEDS", "PREVHYP"
            ] + list(age_splines.columns)
            
            X_input = pd.concat([clean_df, age_splines], axis=1)[final_features]

            # Predict and Calibrate
            raw_risk_score = rsf.predict(X_input)
            predict_df = pd.DataFrame({"rsf_score": raw_risk_score})

            # Survival Platt Scaler outputs the full calibrated survival curve over time
            calibrated_surv_func = calibrator.predict_survival_function(predict_df)
            
            # Fix for calibration layer
            timeline_array = calibrated_surv_func.index.to_numpy()
            closest_time_idx = np.abs(timeline_array - target_days).argmin()
            
            # Extract the clean baseline scalar float
            calibrated_survival_prob = calibrated_surv_func.iloc[:, 0].to_numpy()[closest_time_idx]
            base_risk_pct = (1.0 - calibrated_survival_prob) * 100

            
            impacts = []

            improvements = {
                "SYSBP": {"type": "upper", "max_healthy": 120.0, "target_val": 120.0, "label": "Bringing your systolic blood pressure/top blood pressure number down (under 120)"},
                "TOTCHOL": {"type": "upper", "max_healthy": 180.0, "target_val": 180.0, "label": "Bringing your total cholesterol down (under 180)"},
                "CURSMOKE": {"type": "upper", "max_healthy": 0.0, "target_val": 0.0, "label": "Quitting smoking"},
                "BMI": {"type": "range", "min_healthy": 18.5, "max_healthy": 24.9, "target_high": 24.9, "target_low": 18.5, "label_high": "Reducing weight to a healthy number (under 25 BMI)", "label_low": "Improving nutrition to achieve a healthy weight (over 18.5 BMI)"},
                "GLUCOSE": {"type": "range", "min_healthy": 70.0, "max_healthy": 99.0, "target_high": 99.0, "target_low": 70.0, "label_high": "Bringing blood sugar down into the normal zone (under 100)", "label_low": "Stabilizing low blood sugar into the normal zone (over 70)"}
            }
            
            for feat, info in improvements.items():
                current_val = clean_df[feat].iloc[0] # Grab raw scalar value
                should_simulate = False
                direction = None
                sim_target = None

                # Evaluate standard upper limits
                if info["type"] == "upper" and current_val > info["max_healthy"]:
                    should_simulate = True
                    direction = "lower"
                    sim_target = info["target_val"]
                    display_label = info["label"]    
                # Evaluate target range (BMI & Glucose)
                elif info["type"] == "range":
                    if current_val > info["max_healthy"]:
                        should_simulate = True
                        direction = "lower"
                        sim_target = info["target_high"] # Simulate moving to the upper healthy boundary
                        display_label = info["label_high"]
                    elif current_val < info["min_healthy"]:
                        should_simulate = True
                        direction = "raise"
                        sim_target = info["target_low"]  # Simulate moving up to the lower healthy boundary
                        display_label = info["label_low"]

                if should_simulate:
                    sim_df = clean_df.copy()
                    sim_df[feat] = sim_target # Use the dynamic boundary target
                    
                    # Dynamic re-calculations as usual
                    if feat == "GLUCOSE" and direction == "lower":
                        sim_df["DIABETES"] = 0.0
                    if feat == "SYSBP":
                        sim_df["MAP"] = sim_df["DIABP"] + (1/3) * (sim_df["SYSBP"] - sim_df["DIABP"])
                    
                    sim_df["LOG_TOTCHOL"] = np.log1p(sim_df["TOTCHOL"])
                    sim_df["LOG_CIGS"] = np.log1p(sim_df["CIGPDAY"])
                    
                    X_sim = pd.concat([sim_df, age_splines], axis=1)[final_features]

                    # Predict raw score for the hypothetical scenario and scale it via the Cox wrapper
                    raw_sim_score = rsf.predict(X_sim)
                    predict_sim_df = pd.DataFrame({"rsf_score": raw_sim_score}, index=[0])
                    calibrated_sim_surv = calibrator.predict_survival_function(predict_sim_df)
                    
                    sim_timeline_array = calibrated_sim_surv.index.to_numpy()
                    sim_closest_idx = np.abs(sim_timeline_array - target_days).argmin()
                    
                    sim_survival_prob = calibrated_sim_surv.iloc[:, 0].to_numpy()[sim_closest_idx]
                    sim_risk = (1.0 - sim_survival_prob) * 100

                    reduction = base_risk_pct - sim_risk if direction == "lower" else 0.0
                    
                    # Adjust the difference required to start displaying a recommendation
                    if reduction > 0.1:
                        impacts.append({"label": display_label, "reduction": reduction, "direction": direction})

            return {
                "calibrated_surv_func": calibrated_surv_func, # Passes the calibrated curve downstream for plotting
                "calibrated_risk": base_risk_pct, 
                "imputed": imputed_list,
                "user_vals": user_vals,
                "guidance": sorted(impacts, key=lambda x: x["reduction"], reverse=True) 
            }
        except Exception as err:
            print("An error occurred while processing user inputs.")
            raise err


    @render.ui
    def risk_badge():
        res = process_data()
        if res is None: return ui.p("Adjust factors and click 'Analyze Risk Factors'.")
        
        risk_pct = round(res["calibrated_risk"])

        # Official AHA/ACC Risk Categories
        # Slight adjustment since the low ranges are closer where rounding the values can change the displayed risk.
        # ONLY displays Yellow/Borderline at 5.5% (raw) or above since it's contradictory to say excellent health (Health Guidance) and Borderline.
        if risk_pct >= 20:
            color, label = "#e74c3c", "High Risk"
        elif risk_pct >= 7.5:
            color, label = "#f39c12", "Intermediate Risk"
        elif risk_pct > 5:
            color, label = "#f1c40f", "Borderline Risk"
        else:
            color, label = "#27ae60", "Low Risk"
        
        return ui.div(
            ui.h3(f"{risk_pct}%"),
            ui.p(label, style="font-weight: bold; font-size: 1.2em; margin-bottom: 0;"),
            ui.p("10-Year Cardiovascular Risk", style="font-size: 0.9em; opacity: 0.9;"),
            style=f"padding: 20px; color: white; background-color: {color}; border-radius: 12px; text-align: center; margin-bottom: 15px;"
        )

    @render.plot
    def survival_plot():
        res = process_data()
        if res is None: return None
        
        calibrated_curve = res["calibrated_surv_func"]
        
        # Extract baseline risk percentage
        risk_pct = round(res["calibrated_risk"])
        
        # Synchronize colors with the clinical thresholds
        if risk_pct >= 20:
            color = "#e74c3c" # Red (High)
        elif risk_pct >= 7.5:
            color = "#f39c12" # Orange (Intermediate)
        elif risk_pct > 5:
            color = "#f1c40f" # Yellow (Borderline)
        else:
            color = "#27ae60" # Green (Low)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        
        # Converter matches days to operational year timelines
        timeline_years = calibrated_curve.index.to_numpy() / 365.25
        survival_probabilities = calibrated_curve.iloc[:, 0].to_numpy()
        
        # Instead of survival, we're plotting upwards for event chance over time
        event_chance = 100 - (survival_probabilities * 100)
        
        # Plot the smooth continuous line using the dynamic clinical color (from green to red)
        ax.plot(timeline_years, event_chance, color=color, lw=3)
        
        # Focus on 10 year span
        ax.set_xlim(0, 10)
        ax.set_xticks(range(11))
        ax.set_ylim(0, 100)
        
        # Clean up graph borders 
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#bdc3c7")
        ax.spines["bottom"].set_color("#bdc3c7")
        ax.grid(axis="y", linestyle=":", alpha=0.5, color="#bdc3c7")

        ax.set_xlabel("Years into the Future", fontsize=11, color="#34495e", labelpad=10)
        ax.set_ylabel("Accumulated Chance of an Event (%)", fontsize=11, color="#34495e", labelpad=10)
        ax.tick_params(colors="#7f8c8d", labelsize=10)
        
        plt.title(
            f"Your Risk Gradually Rises to {risk_pct}% Within 10 Years", 
            fontsize=13, fontweight="bold", pad=15, color="#2c3e50", loc="left"
        )
        
        # Soft background tint to fill the risk area beneath the calibrated line
        ax.fill_between(timeline_years, event_chance, color=color, alpha=0.08)
        
        return fig


    @render.plot
    def isotype_plot():
        res = process_data()
        if res is None: return None
        
        risk_pct = round(res["calibrated_risk"])
        
        # Force at least 1 colored silhouette instead of snapping awkwardly to 0
        fill_count = int(np.ceil(risk_pct / 10.0))
        
        # Prevent edge case errors if risk is exactly 0% or above 100%
        fill_count = max(0, min(10, fill_count))
        
        # Define colors based on existing risk scheme (should've extracted this and reused when needed)
        if risk_pct >= 20: color = "#e74c3c"
        elif risk_pct >= 7.5: color = "#f39c12"
        elif risk_pct > 5: color = "#f1c40f"
        else: color = "#27ae60"

        # Define a clean Human Icon Path (Broad shoulders, rounded head). 
        verts = [
            (0.5, 1.0), (0.7, 0.95), (0.7, 0.8), (0.5, 0.75), # Head
            (0.9, 0.7), (0.9, 0.05), (0.1, 0.05), (0.1, 0.7), # Body
            (0.5, 0.75), (0.3, 0.8), (0.3, 0.95), (0.5, 1.0), # Head back
        ]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3,
                Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
                Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
        person_path = Path(verts, codes)

        fig, ax = plt.subplots(figsize=(10, 1.3))
        ax.set_xlim(0, 10)
        ax.set_ylim(0.0, 1.05) 
        ax.axis("off")

        for i in range(10):
            trans = mtransforms.Affine2D().translate(i, 0) + ax.transData
            
            # Background 
            ax.add_patch(patches.PathPatch(person_path, facecolor="#ecf0f1", lw=0, transform=trans))
            
            # We calculate the fill for each person
            this_fill = max(0, min(1, fill_count - i))
            
            if this_fill > 0:
                # Create a vertical gradient (Fading from solid color to transparent/gray)
                grad_map = LinearSegmentedColormap.from_list("risk_grad", [color, "#ecf0f1"])
                
                # The 'partial' person gets a gradient, the 'full' people stay solid
                is_partial = 0 < this_fill < 1
                current_color = color if not is_partial else grad_map(0.3) # Slightly faded if partial
                
                # Clipping mask for the silhouette
                clip_rect = patches.Rectangle((i, 0), 1, this_fill, transform=ax.transData)
                
                fill_patch = patches.PathPatch(person_path, facecolor=current_color, 
                                            alpha=0.85 if is_partial else 1.0, 
                                            lw=0, transform=trans)
                fill_patch.set_clip_path(clip_rect)
                ax.add_patch(fill_patch)

        plt.title(f"10-Year Probability: {risk_pct:.0f}%", fontsize=12, fontweight='bold', pad=4, color="#34495e", loc="left")
        plt.tight_layout(pad=0.2)
        return fig


    @render.text
    def formatted_inputs():
        res = process_data()
        if not res: 
            return "No data analyzed yet. Please click 'Analyze Risk Factors' first."
        
        vals = res["user_vals"]
        # Using raw risk instead of the rounded in the visuals. Can add round() to match behavior.
        risk = res["calibrated_risk"]
        
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
        # If rounding the risk, adjust the output here too
        summary_header = (
            "--- PATIENT DATA SUMMARY ---\n"
            f"10-YEAR ABSOLUTE RISK (UNROUNDED): {risk:.3f}%\n"
            "----------------------------\n"
        )
        return summary_header + "\n".join(lines)


    @render.ui
    def log_display():
        return ui.markdown(f"**Log Status:** `{error_log}`")

    @render.ui
    def imputation_alert():
        res = process_data()
        if res and res["imputed"]:
            cols = ", ".join(res["imputed"])
            return ui.div(f"ℹ️ Note: Missing values for ({cols}) were estimated programmatically. Fill in all known fields for more accurate results.", 
                          style="color: #856404; background-color: #fff3cd; padding: 12px; border-radius: 6px; margin-top: 10px;")

    @render.ui
    def rec_list():
        res = process_data() 
        if res is None:
            return ui.p("Please adjust your metrics and click 'Analyze Risk Factors' on the sidebar to view guidance.")
        
        impacts = res["guidance"][:3]
        
        if not impacts:
            return ui.div(
                ui.h4("✨ Excellent Progress!"),
                ui.p("Based on the predictive model, your modifiable metrics (Blood Pressure, Cholesterol, BMI, Smoking status) are already close to or at clinical targets."),
                ui.p("Continue focusing on maintaining a heart-healthy diet, staying physically active, and attending regular wellness checkups."),
                style="padding: 15px; background-color: #f1f9f5; border-left: 5px solid #27ae60; border-radius: 4px;"
            )

        # Actionable tips
        clinical_tips = {
            "SYSBP": [
                "Fill half your plate with vegetables, fruits, and whole grains.",
                "Cut back on red meat, butter, and sweets.",
                "Avoid packaged foods, canned soups, and added salt.",
                "Walk briskly, cycle, or swim for 30 minutes daily."
            ],
            "TOTCHOL": [
                "Eat more oatmeal, beans, apples, and berries.",
                "Choose lean meats such as chicken or fish instead of red meats such as beef or pork.",
                "Cook with olive oil instead of butter or lard.",
                "Avoid fried foods."
            ],
            "CURSMOKE": [
                "Quit smoking completely to protect your blood vessels.",
                "Use nicotine patches or gum to stop physical cravings.",
                "Throw away ashtrays, lighters, and avoid smoking areas."
            ],
            "BMI_LOWER": [
                "Use smaller plates to naturally shrink your portions.",
                "Swap chips and sodas for nuts, fresh fruit, and water.",
                "Do light exercises like squats to burn energy naturally."
            ],
            "BMI_RAISE": [
                "Add avocados, walnuts, peanut butter, and olive oil to meals.",
                "Eat 5 or 6 smaller meals throughout the day."
            ],
            "GLUCOSE_LOWER": [
                "Replace soda, sweet tea, and juice with plain water.",
                "Swap white bread and rice for oatmeal and brown rice.",
                "Take a short walk right after your meals."
            ],
            "GLUCOSE_RAISE": [
                "Eat lean meat and healthy carbs every few hours.",
                "Carry a piece of fruit or juice in case you feel shaky."
            ]
        }
            
        list_items = []
        for item in impacts:
            # Standardize direction suffix to match new dict keys (_LOWER or _RAISE)
            direction_suffix = f"_{item.get('direction', 'lower').upper()}" if "direction" in item else ""
            if direction_suffix == "_LOWER":
                direction_suffix = "_LOWER"
            elif direction_suffix == "_RAISE":
                direction_suffix = "_RAISE"
                
            # Convert the label to lowercase for safe matching
            label_lower = item["label"].lower()
            
            feature_key = None
            if "blood pressure" in label_lower: 
                feature_key = "SYSBP"
            elif "cholesterol" in label_lower: 
                feature_key = "TOTCHOL"
            elif "smoking" in label_lower: 
                feature_key = "CURSMOKE"
            elif "bmi" in label_lower or "weight" in label_lower: 
                feature_key = "BMI" + direction_suffix
            elif "blood sugar" in label_lower or "blood sugar" in label_lower or "normal zone" in label_lower: 
                feature_key = "GLUCOSE" + direction_suffix

            # Build the action sub-list
            tip_elements = []
            if feature_key and feature_key in clinical_tips:
                for tip in clinical_tips[feature_key]:
                    tip_elements.append(ui.tags.li(tip, style="font-size: 0.95em; color: #555; list-style-type: circle;"))

            # Compile the recommendation block
            list_items.append(
                ui.tags.li(
                    ui.div(
                        ui.markdown(f"**{item['label']}**: Could reduce your overall 10-year risk by **{item['reduction']:.1f}%**"),
                        ui.tags.ul(*tip_elements, style="margin-top: 5px; margin-bottom: 15px; padding-left: 20px;") if tip_elements else None
                    ),
                    style="list-style-type: square; margin-bottom: 10px;"
                )
            )
            
        return ui.div(
            ui.h4("Your Personalized Improvement Opportunities:"),
            ui.p("These lifestyle changes would improve your 10 year outlook most:"),
            ui.tags.ul(*list_items, style="font-size: 1.1em; line-height: 1.65; margin-top: 15px;"),
            ui.hr(),
            ui.p("💡 *These calculations represent statistical simulations generated directly by the underlying Random Survival Forest based on your unique patient profile. Actual clinical intervention should always be managed by a physician.*", 
                style="font-size: 0.85em; color: #7f8c8d; line-height: 1.4;")
        )



app = App(app_ui, server)
