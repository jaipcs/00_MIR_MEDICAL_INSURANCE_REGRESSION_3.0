import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Page Configuration ------------------
st.set_page_config(page_title="Insurance Expenses Predictor", page_icon="💸", layout="wide")
st.title("💰 Insurance Expenses Prediction App")
st.markdown("Upload a CSV and predict medical expenses using the best trained model.")

# ------------------ File Upload ------------------
st.sidebar.header("📂 Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Upload CSV for Prediction + EDA", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # ------------------ EDA Summary ------------------
    st.subheader("🧪 Automated EDA Summary")
    eda_text = ""

    shape_info = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
    na_info = df.isnull().sum()
    dup_info = df.duplicated().sum()
    numeric_corr = df.corr(numeric_only=True)

    eda_text += f"**Shape**: {shape_info}\n"
    eda_text += f"**Duplicates**: {dup_info}\n"
    eda_text += "**Missing Values**:\n"
    eda_text += f"{na_info[na_info > 0] if na_info.sum() > 0 else 'None'}\n\n"
    if "expenses" in df.columns:
        eda_text += "**Correlation with Target (`expenses`)**:\n"
        eda_text += f"{numeric_corr['expenses'].sort_values(key=lambda s: s.abs(), ascending=False)}\n"
    else:
        eda_text += "_Column `expenses` not found for correlation._\n"

    st.text(eda_text)

    # Show histogram
    st.subheader("📈 Numeric Distributions")
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

# ------------------ Sidebar Input ------------------
st.sidebar.header("🔧 Manual Input for Prediction")
age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 0)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

input_df = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex": sex,
    "smoker": smoker,
    "region": region
}])

st.subheader("🔍 Input Preview")
st.dataframe(input_df)

# ------------------ Load Best Model ------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

try:
    model = load_model()
    if st.button("🎯 Predict Medical Expenses"):
        prediction = model.predict(input_df)[0]
        st.success(f"💡 Estimated Medical Expenses: **${prediction:,.2f}**")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# ------------------ Show Model Comparison ------------------
if os.path.exists("model_metrics_comparison_ranked.csv"):
    st.subheader("📋 Ranked Model Metrics")
    df_metrics = pd.read_csv("model_metrics_comparison_ranked.csv")
    st.dataframe(df_metrics)

# ------------------ Fit Ranking Summary ------------------
if os.path.exists("model_fit_ranking.csv"):
    st.subheader("🏆 Model Fit Status Summary")
    fit_df = pd.read_csv("model_fit_ranking.csv")
    st.dataframe(fit_df)