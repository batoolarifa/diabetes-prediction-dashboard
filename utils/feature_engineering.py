
import pandas as pd
import numpy as np

def create_features(df):
    """Generate all engineered features for the diabetes model."""
    df = df.copy()
    
    df["daily_physical_hours"] = df["physical_activity_minutes_per_week"] / 60 / 7
    df["screen_activity_ratio"] = df["screen_time_hours_per_day"] / (df["daily_physical_hours"] + 1e-6)
    df["sleep_efficiency_pct"] = df["sleep_hours_per_day"] / (24 - df["screen_time_hours_per_day"] - df["daily_physical_hours"] + 1e-6)
    df["activity_x_age"] = df["physical_activity_minutes_per_week"] * df["age"]
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["pulse_pressure_ratio"] = df["pulse_pressure"] / df["systolic_bp"]
    df["mean_arterial_pressure"] = df["diastolic_bp"] + df["pulse_pressure"] / 3
    df["rate_pressure_product"] = df["heart_rate"] * df["systolic_bp"]
    df["bp_ratio"] = df["systolic_bp"] / (df["diastolic_bp"] + 1e-6)
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1e-6)
    df["cholesterol_hdl_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-6)
    df["non_hdl_cholesterol"] = df["cholesterol_total"] - df["hdl_cholesterol"]
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-6)
    df["lipid_burden"] = df["ldl_hdl_ratio"] + df["tg_hdl_ratio"] + df["cholesterol_hdl_ratio"]
    df["chol_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-6)
    df["age_bmi_risk"] = df["age"] * df["bmi"]
    df["age_norm_activity"] = df["physical_activity_minutes_per_week"] / (df["age"] + 1)
    df["bmi_waist_ratio"] = df["bmi"] * df["waist_to_hip_ratio"]
    df["metabolic_risk"] = df["bmi"] * df["ldl_hdl_ratio"]
    df["bmi_age"] = df["bmi"] * df["age"]
    df["risk_history"] = df["hypertension_history"] + df["cardiovascular_history"]
    df["genetic_history"] = df["family_history_diabetes"] * df["bmi"]
    df["af_risk"] = df["family_history_diabetes"] + df["family_history_diabetes"] * df["age"] * 0.15
    df["at_risk"] = df["triglycerides"] + df["triglycerides"] * df["age"] * 0.3
    df["activity_screen_ratio"] = df["physical_activity_minutes_per_week"] / (df["screen_time_hours_per_day"] + 1)
    
    return df
