import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")


@st.cache_resource
def load_artifacts():
    return (
        joblib.load("best_model.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("feature_columns.pkl"),
        joblib.load("encoder.pkl"),
    )


model, scaler, feature_columns, encoder = load_artifacts()

if isinstance(model, str):
    st.error(
        "best_model.pkl contains only the text '" + model + "', not a trained model. "
        "Re-save the trained Random Forest in Colab (joblib.dump(rf, 'best_model.pkl')) "
        "and upload the new file."
    )
    st.stop()

# Dropdown options come from the fitted encoder
gender_options, smoking_options = [list(c) for c in encoder.categories_]

st.title("🩺 Diabetes Prediction")
st.caption("Random Forest classifier · trained on the Kaggle diabetes prediction dataset")

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", gender_options)
    age = st.number_input("Age (years)", 0.08, 120.0, 40.0, step=1.0)
    bmi = st.number_input("BMI", 10.0, 100.0, 27.0, step=0.1)
    smoking = st.selectbox("Smoking history", smoking_options)
with c2:
    hba1c = st.number_input("HbA1c level (%)", 3.0, 15.0, 5.5, step=0.1)
    glucose = st.number_input("Blood glucose level (mg/dL)", 50, 400, 120, step=1)
    hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
    heart = st.radio("Heart disease", ["No", "Yes"], horizontal=True)


def build_row():
    row = {c: 0 for c in feature_columns}
    row.update(
        {
            "age": age,
            "hypertension": int(hypertension == "Yes"),
            "heart_disease": int(heart == "Yes"),
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose,
        }
    )
    # One-hot columns (the first category of each was dropped in training)
    for col in (f"gender_{gender}", f"smoking_history_{smoking}"):
        if col in row:
            row[col] = 1
    return pd.DataFrame([row])[feature_columns]


if st.button("Predict", type="primary"):
    X = scaler.transform(build_row())
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    if pred == 1:
        st.error(f"⚠️ Likely **diabetic** — probability {prob:.1%}")
    else:
        st.success(f"✅ Likely **not diabetic** — probability of diabetes {prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption("Educational project only — not a medical diagnosis.")

with st.expander("About this model"):
    st.write(
        "Compared Logistic Regression, KNN, Decision Tree, Random Forest and SVM on 100,000 "
        "records. Random Forest scored best (accuracy ≈ 97.0%, precision 0.95, recall 0.69 "
        "for the diabetic class)."
    )
