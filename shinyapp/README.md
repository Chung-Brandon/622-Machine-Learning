---
title: Heart Health Risks and Recommendations
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Heart Health Risks and Recommendations

This project uses a **Random Survival Forest (RSF)** model with Platt calibration trained on the Framingham Heart Study to predict the 10-year risk of Cardiovascular Disease (CVD).

## 🚀 How to Run Locally

If you want to run this model on your own machine, follow these steps:
```bash
python -m venv .venv
.venv\Scripts\activate 
pip install -r requirements.txt
shiny run --reload --launch-browser app.py