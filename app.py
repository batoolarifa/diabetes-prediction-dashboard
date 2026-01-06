import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from utils.feature_engineering import create_features
from models.predict_model import load_model

model = load_model()
model_features = model.feature_name_

st.set_page_config(
    page_title="💉 Diabetes Risk Dashboard",
    page_icon="💉",
    layout="wide"
)

st.markdown("""
<style>
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSlider"] input,
div[data-testid="stSelectbox"] div[role="combobox"] {
    color: #fffff !important;               
    

div[data-testid="stSelectbox"] div[role="option"] {
    color: #000 !important;
}

div[data-testid="stNumberInput"] input::placeholder,
div[data-testid="stTextInput"] input::placeholder {
    color: #555 !important;
}

button[kind="primary"] {
    border-radius: 10px;
    height: 3em;
    background-color: #FFD700;  /* gold */
    color: #000;
}

.risk-high { color: #FF4500; font-weight: bold; }      
.risk-moderate { color: #FFD700; font-weight: bold; }  
.risk-low { color: #87CEFA; font-weight: bold; }       
</style>

""", unsafe_allow_html=True)

st.title("💉 Diabetes Risk Assessment Dashboard")
st.markdown("Estimate your **risk of diabetes** using clinical and lifestyle indicators.")

st.markdown("---")

# Patient Inputs
st.header("🧑‍⚕️ Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    age = st.number_input("Age (years)", 1, 120, 30, step=1)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0, step=0.1)
    waist_to_hip_ratio = st.number_input("Waist-to-Hip Ratio", 0.3, 1.5, 0.85, step=0.01)
    physical_activity = st.number_input("Physical Activity (minutes/week)", 0, 1000, 150)

with col2:
    st.subheader("Vitals")
    sleep_hours = st.number_input("Sleep Hours / Day", 0, 24, 7)
    systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", 50, 130, 80)
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 70)

with col3:
    st.subheader("Medical History")
    family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])
    cardio_history = st.selectbox("Cardiovascular Disease History", ["No", "Yes"])
    htn_history = st.selectbox("Hypertension History", ["No", "Yes"])

# optional labs 
with st.expander("🧪 Optional Lab Measurements"):
    chol_total = st.number_input("Total Cholesterol", 50, 400, 180)
    hdl_chol = st.number_input("HDL Cholesterol", 10, 150, 50)
    ldl_chol = st.number_input("LDL Cholesterol", 10, 200, 100)
    triglycerides = st.number_input("Triglycerides", 10, 500, 120)

# validation warnings
if bmi > 45: st.warning("⚠️ BMI is extremely high. Verify value.")
if systolic_bp > 180: st.warning("⚠️ Systolic BP is high.")

st.markdown("---")

# prepare input data
input_df = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "waist_to_hip_ratio": waist_to_hip_ratio,
    "physical_activity_minutes_per_week": physical_activity,
    "screen_time_hours_per_day": 0,
    "sleep_hours_per_day": sleep_hours,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "heart_rate": heart_rate,
    "cholesterol_total": chol_total if 'chol_total' in locals() else 0,
    "hdl_cholesterol": hdl_chol if 'hdl_chol' in locals() else 0,
    "ldl_cholesterol": ldl_chol if 'ldl_chol' in locals() else 0,
    "triglycerides": triglycerides if 'triglycerides' in locals() else 0,
    "family_history_diabetes": 1 if family_history == "Yes" else 0,
    "cardiovascular_history": 1 if cardio_history == "Yes" else 0,
    "hypertension_history": 1 if htn_history == "Yes" else 0
}])

# prediction
st.header("📊 Risk Prediction")
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
    with st.spinner("Analyzing patient data..."):
        df_features = create_features(input_df)
        
        for f in model_features:
            if f not in df_features.columns:
                df_features[f] = 0
        
        df_features = df_features[model_features]
        prob = model.predict_proba(df_features)[:, 1][0]

    # Risk Color Logic
    if prob >= 0.7:
        risk_label = "High Risk"
        color = "#FF4500"  
    elif prob >= 0.4:
        risk_label = "Moderate Risk"
        color = "#FFD700"
    else:
        risk_label = "Low Risk"
        color = "#87CEFA"  

    
    # patient summary
    st.subheader("🧾 Patient Summary")
    st.info(
        f"""
        **Age:** {age} years  
        **BMI:** {bmi}  
        **Blood Pressure:** {systolic_bp}/{diastolic_bp} mmHg  
        **Physical Activity:** {physical_activity} min/week  
        """
    )

    # risk visualization
    st.subheader("📈 Diabetes Risk Score")
    st.progress(int(prob * 100))
    st.markdown(f"<h3 style='color:{color}'>{risk_label}</h3>", unsafe_allow_html=True)

    # recommendations
    st.subheader("🩺 Recommended Actions")
    if prob >= 0.7:
        st.error("- Consult an endocrinologist\n- Begin blood glucose monitoring\n- Reduce sugar & refined carbs\n- Increase physical activity")
    elif prob >= 0.4:
        st.warning("- Improve diet quality\n- Increase daily movement\n- Monitor BMI and BP regularly")
    else:
        st.success("- Maintain healthy lifestyle\n- Routine screening annually")


    # explainability top features
    st.subheader("🔍 Top 5 Features Influencing Prediction")
    importances = model.feature_importances_
    features = df_features.columns
    values = df_features.iloc[0].values
    contributions = pd.DataFrame({
        "Feature": features,
        "Contribution": importances * values
    })
    top5 = contributions.reindex(contributions["Contribution"].abs().sort_values(ascending=False).index).head(5)

    fig = px.bar(
        top5[::-1],
        x="Contribution",
        y="Feature",
        orientation="h",
        color="Contribution",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        text="Contribution",
        title="Top 5 Factors Influencing Prediction"
    )
    fig.update_layout(template="plotly_white", height=400, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # download input data
    st.download_button(
        "💾 Download Patient Input Data",
        input_df.to_csv(index=False),
        file_name="patient_input.csv",
        mime="text/csv"
    )


# footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #555; font-size: 14px; padding: 10px 0;">
💉 Diabetes Risk Dashboard • Built with Streamlit • Version 1.0
</div>
""", unsafe_allow_html=True)

