import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# App title and description
st.title("❤️ Heart Disease Risk Assessment")
st.markdown("""
**Clinical Decision Support Tool**  
Predicts heart disease risk based on patient health metrics.  
For nursing staff use only.
Note: This tool is for educational purposes and should not replace clinical judgment. 
""")

# Define options with clinical terminology

gen_health_options = ["Poor", "Fair", "Good", "Very good", "Excellent"]
diabetic_options = ["No", "Yes",
                    "No, borderline diabetes", "Yes (during pregnancy)"]
age_categories = [
    '18-24', '25-29', '30-34', '35-39', '40-44',
    '45-49', '50-54', '55-59', '60-64', '65-69',
    '70-74', '75-79', '80 or older'
]

# Create input form
with st.form("clinical_assessment_form"):
    st.header("Patient Health Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        age_category = st.selectbox("Age Group", age_categories)
        sex = st.radio("Sex", ["Female", "Male"])

        st.subheader("Vital Signs & Metrics")
        bmi = st.slider("BMI", 10.0, 50.0, 25.0, step=0.1,
                        help="Body Mass Index")
        physical_health = st.slider(
            "Physical health issues (past 30 days)", 0, 30, 0,
            help="Days with physical health problems")
        mental_health = st.slider(
            "Mental health issues (past 30 days)", 0, 30, 0,
            help="Days with mental health problems")
        sleep_time = st.slider("Avg. daily sleep (hours)", 1, 12, 7)

    with col2:
        st.subheader("Health History")
        stroke = st.radio("History of stroke?", ["No", "Yes"])
        diff_walking = st.radio(
            "Difficulty walking/climbing stairs?", ["No", "Yes"])
        asthma = st.radio("Asthma diagnosis?", ["No", "Yes"])
        kidney_disease = st.radio("Kidney disease?", ["No", "Yes"])
        skin_cancer = st.radio("Skin cancer?", ["No", "Yes"])
        diabetic = st.selectbox("Diabetes status", diabetic_options)
        gen_health = st.selectbox("General health rating", gen_health_options)

        st.subheader("Lifestyle Factors")
        smoking = st.radio("Current smoker?", ["No", "Yes"])
        alcohol = st.radio("Heavy alcohol consumption?", ["No", "Yes"],
                           help=">14 drinks/week for men, >7 for women")
        physical_activity = st.radio("Regular physical activity?", ["No", "Yes"],
                                     help=">150 mins moderate or >75 mins vigorous/week")

    submitted = st.form_submit_button("Assess Heart Disease Risk")

if submitted:
    try:
        # Create input dictionary
        input_data = {
            'BMI': [bmi],
            'Smoking': [1 if smoking == "Yes" else 0],
            'AlcoholDrinking': [1 if alcohol == "Yes" else 0],
            'Stroke': [1 if stroke == "Yes" else 0],
            'PhysicalHealth': [physical_health],
            'MentalHealth': [mental_health],
            'DiffWalking': [1 if diff_walking == "Yes" else 0],
            'Sex': [1 if sex == "Male" else 0],
            'AgeCategory': [age_categories.index(age_category)],
            'Diabetic': [diabetic_options.index(diabetic)],
            'PhysicalActivity': [1 if physical_activity == "Yes" else 0],
            'GenHealth': [gen_health_options.index(gen_health)],
            'SleepTime': [sleep_time],
            'Asthma': [1 if asthma == "Yes" else 0],
            'KidneyDisease': [1 if kidney_disease == "Yes" else 0],
            'SkinCancer': [1 if skin_cancer == "Yes" else 0],

        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # Apply scaling
        features_to_scale = ['BMI', 'PhysicalHealth',
                             'MentalHealth', 'SleepTime']
        input_df[features_to_scale] = scaler.transform(
            input_df[features_to_scale])

        # Make prediction - CHANGED TO CLASS PREDICTION
        # Get class prediction (0 or 1)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(
            input_df)[0][1]  # For confidence score
        risk_percentage = round(probability * 100, 1)

        # Display results - UPDATED TO SHOW CLASS PREDICTION
        st.header("Risk Assessment Results")
        result_col, confidence_col = st.columns(2)

        with result_col:
            st.subheader("Clinical Prediction")
            if prediction == 1:  # 1 indicates "At risk"
                st.error("**At Risk of Heart Disease**")
            else:
                st.success("**Not At Risk of Heart Disease**")

        with confidence_col:
            st.subheader("Model Confidence")
            st.markdown(f"<h1 style='text-align: center;'>{risk_percentage}%</h1>",
                        unsafe_allow_html=True)

        # Visual risk indicator
        st.progress(probability)

        # Interpretation and clinical guidance
        st.subheader("Clinical Guidance")
        if probability >= 0.5:
            st.warning("""
            **This patient shows elevated risk for heart disease.**
            Recommended actions:
            - Schedule cardiology consultation within 2 weeks
            - Order ECG and lipid profile tests
            - Provide heart health education materials
            - Schedule follow-up appointment in 1 month
            """)
        else:
            st.info("""
            **This patient shows low risk for heart disease.**
            Recommended actions:
            - Reinforce heart-healthy lifestyle habits
            - Provide preventive care education
            - Schedule annual wellness check
            """)

        # Risk factor summary
        st.subheader("Key Risk Factors")
        risk_factors = []
        if smoking == "Yes":
            risk_factors.append("Smoking")
        if bmi >= 30:
            risk_factors.append(f"Obesity (BMI: {bmi})")
        if physical_health >= 15:
            risk_factors.append("Frequent physical health issues")
        if age_category in ['55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']:
            risk_factors.append(f"Age ({age_category})")
        if diabetic != "No":
            risk_factors.append("Diabetes")

        if risk_factors:
            st.write("Identified risk factors: " + ", ".join(risk_factors))
        else:
            st.write("No major risk factors identified")

    except Exception as e:
        st.error(f"Assessment failed: Please check all fields are complete")
