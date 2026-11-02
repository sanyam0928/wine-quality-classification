import os
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="WineWise | Quality Predictor", page_icon="🍷", layout="wide")

# -------------------- THEME --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
.stApp { background: linear-gradient(135deg, #14090f 0%, #24101b 55%, #170d13 100%); color: #f8edf0; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #fff4f0 !important; }
p, label, div, span { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #1c0d15; border-right: 1px solid #573044; }
[data-testid="stMetric"] { background: #2a1420; border: 1px solid #704052; padding: 16px; border-radius: 14px; }
.stButton > button { background: #a83255; color: white; border: 0; border-radius: 10px; font-weight: 700; }
.stButton > button:hover { background: #c44b6c; color: white; }
.hero { padding: 30px; border: 1px solid #704052; border-radius: 20px; background: linear-gradient(120deg, #351626, #21111b); }
.small-note { color: #d9b9c4; font-size: 0.92rem; }
div[data-baseweb="input"] input { background: #fff8fa; color: #24101b; }
</style>
""", unsafe_allow_html=True)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_FILE = "winequality-red.csv"
FEATURES = ["alcohol", "volatile acidity", "ph", "sulphates"]

@st.cache_data(show_spinner=False)
def download_dataset():
    if not os.path.exists(DATA_FILE):
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    return pd.read_csv(DATA_FILE, sep=";")

def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    needed = FEATURES + ["quality"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing these required columns: " + ", ".join(missing))
    df = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
    df["quality_category"] = pd.cut(
        df["quality"], bins=[-float("inf"), 4, 6, float("inf")],
        labels=["Low", "Medium", "High"]
    )
    df = df.dropna(subset=["quality_category"])
    return df

@st.cache_resource(show_spinner="Training the Random Forest model...")
def train_model(df):
    X = df[FEATURES]
    y = df["quality_category"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=None, random_state=42,
            class_weight="balanced", min_samples_split=4
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, X_test, y_test, predictions, accuracy

def style_chart(fig):
    fig.patch.set_facecolor("#2a1420")
    for ax in fig.axes:
        ax.set_facecolor("#2a1420")
        ax.tick_params(colors="#f5e7ec")
        ax.xaxis.label.set_color("#f5e7ec")
        ax.yaxis.label.set_color("#f5e7ec")
        ax.title.set_color("#fff4f0")
        for spine in ax.spines.values():
            spine.set_color("#704052")
    return fig

# -------------------- LOAD DATA --------------------
st.sidebar.markdown("# 🍷 WineWise")
st.sidebar.caption("A machine-learning wine quality project")
page = st.sidebar.radio("NAVIGATION", ["Home", "Predict Quality", "Dataset Explorer", "Model Performance", "About Project"])
st.sidebar.markdown("---")
st.sidebar.markdown('<p class="small-note">Built with Python • Streamlit • Scikit-learn</p>', unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Optional: upload your winequality-red.csv", type=["csv"])
try:
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded, sep=None, engine="python")
    else:
        raw_df = download_dataset()
    data = prepare_data(raw_df)
    model, X_test, y_test, predictions, accuracy = train_model(data)
except Exception as e:
    st.error(f"Could not load or process the dataset: {e}")
    st.info("Use the original UCI/Kaggle red wine quality CSV. It should contain columns such as alcohol, volatile acidity, pH, sulphates and quality.")
    st.stop()

# -------------------- PAGES --------------------
if page == "Home":
    st.markdown("""
    <div class="hero">
      <div style="color:#e5a8ba; letter-spacing:3px; font-weight:700;">MACHINE LEARNING • CLASSIFICATION</div>
      <h1 style="font-size:3.1rem; margin:8px 0;">Wine Quality Predictor</h1>
      <p style="font-size:1.12rem; color:#ead5dc; max-width:760px;">
      Explore wine chemistry and predict whether a red wine sample is likely to be
      Low, Medium, or High quality using a Random Forest machine-learning model.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns(3)
    a.metric("Wine samples", f"{len(data):,}")
    b.metric("Model test accuracy", f"{accuracy*100:.1f}%")
    c.metric("Input features", "4")
    st.write("")
    st.subheader("How it works")
    cols = st.columns(4)
    steps = [("01", "Enter chemistry", "Provide four measurable wine properties."),
             ("02", "Model analysis", "Random Forest compares the values with learned patterns."),
             ("03", "Quality category", "The model predicts Low, Medium or High."),
             ("04", "Explore results", "Review charts and model performance.")]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""<div class="hero" style="height:175px;padding:17px;">
            <div style="color:#e5a8ba;font-size:1.5rem;font-weight:700;">{num}</div>
            <h3 style="font-size:1.15rem;">{title}</h3>
            <p class="small-note">{desc}</p></div>""", unsafe_allow_html=True)
    st.info("This is an educational demonstration. It does not replace professional laboratory testing or expert wine evaluation.")
    st.subheader("Dataset preview")
    st.dataframe(data.head(8), use_container_width=True, hide_index=True)

elif page == "Predict Quality":
    st.title("Predict Wine Quality")
    st.write("Enter the chemical measurements below. Example values are pre-filled to help with your first demonstration.")
    st.markdown('<div class="small-note">Use realistic values from the dataset. The prediction is based on four features only.</div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            alcohol = st.number_input("Alcohol (%)", min_value=8.0, max_value=16.0, value=10.5, step=0.1)
            volatile_acidity = st.number_input("Volatile acidity (g/dm³)", min_value=0.10, max_value=2.00, value=0.50, step=0.01)
        with c2:
            ph = st.number_input("pH level", min_value=2.70, max_value=4.20, value=3.30, step=0.01)
            sulphates = st.number_input("Sulphates (g/dm³)", min_value=0.20, max_value=2.00, value=0.65, step=0.01)
        submitted = st.form_submit_button("🍷 Predict Wine Quality", use_container_width=True)
    if submitted:
        sample = pd.DataFrame([[alcohol, volatile_acidity, ph, sulphates]], columns=FEATURES)
        prediction = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        classes = model.named_steps["classifier"].classes_
        confidence = float(max(probabilities))
        color = {"Low":"#e58b8b", "Medium":"#e7c477", "High":"#8ed3a3"}[prediction]
        st.markdown(f"""<div class="hero" style="text-align:center;margin-top:20px;">
        <div style="color:#e5a8ba;letter-spacing:2px;">PREDICTED CATEGORY</div>
        <h1 style="font-size:3rem;color:{color} !important;">{prediction} Quality</h1>
        <p class="small-note">Model confidence: {confidence*100:.1f}%</p>
        </div>""", unsafe_allow_html=True)
        st.write("")
        st.subheader("Your submitted values")
        st.dataframe(sample, use_container_width=True, hide_index=True)
        st.caption("Confidence is the model's estimated class probability, not a guarantee of real-world wine quality.")

elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.write("Understand the dataset used to train the model.")
    a, b, c = st.columns(3)
    a.metric("Total samples", f"{len(data):,}")
    b.metric("Average quality score", f"{data['quality'].mean():.2f}/10")
    c.metric("Unique quality scores", str(data["quality"].nunique()))
    st.subheader("Quality category distribution")
    counts = data["quality_category"].value_counts().reindex(["Low", "Medium", "High"])
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Number of samples in each category")
    ax.set_xlabel("Quality category")
    ax.set_ylabel("Number of wine samples")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(style_chart(fig))
    st.subheader("Alcohol distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(data["alcohol"], bins=20, edgecolor="white")
    ax2.set_title("Distribution of alcohol percentage")
    ax2.set_xlabel("Alcohol (%)")
    ax2.set_ylabel("Number of samples")
    st.pyplot(style_chart(fig2))
    st.subheader("Feature summary")
    st.dataframe(data[FEATURES + ["quality"]].describe().round(2), use_container_width=True)
    st.subheader("Sample records")
    st.dataframe(data.sample(min(15, len(data)), random_state=7), use_container_width=True, hide_index=True)

elif page == "Model Performance":
    st.title("Model Performance")
    st.write("The model is evaluated on a test set that was not used during training.")
    a, b = st.columns(2)
    a.metric("Test accuracy", f"{accuracy*100:.2f}%")
    b.metric("Test samples", f"{len(y_test):,}")
    labels = ["High", "Low", "Medium"]
    cm = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    st.pyplot(style_chart(fig))
    st.subheader("Classification report")
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)
    st.subheader("Which features matter most?")
    rf = model.named_steps["classifier"]
    importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    importance.plot(kind="barh", ax=ax2)
    ax2.set_title("Random Forest feature importance")
    ax2.set_xlabel("Relative importance")
    st.pyplot(style_chart(fig2))
    st.caption("Accuracy and feature importance are calculated from this dataset split. Results can change with different data or model settings.")

else:
    st.title("About the Project")
    st.subheader("Problem statement")
    st.write("Wine quality assessment can involve laboratory measurements and expert evaluation. This project demonstrates how machine learning can learn patterns between selected chemical properties and quality labels.")
    st.subheader("Objective")
    st.write("Build a beginner-friendly web application that classifies red wine samples into Low, Medium and High quality categories using four chemical features.")
    st.subheader("Features used")
    st.table(pd.DataFrame({
        "Feature": FEATURES,
        "Meaning": [
            "Approximate alcohol content of the wine.",
            "A measure related to acetic acid; high values can create an unpleasant vinegar-like taste.",
            "A measure of acidity or alkalinity.",
            "A compound associated with wine preservation and flavour."
        ]
    }))
    st.subheader("Algorithm: Random Forest")
    st.write("Random Forest combines predictions from many decision trees. Each tree learns patterns from the training data, and the forest combines their votes to predict the final category.")
    st.subheader("How categories are created")
    st.markdown("- **Low:** original quality score 4 or below\n- **Medium:** original quality score 5 or 6\n- **High:** original quality score 7 or above")
    st.subheader("Limitations")
    st.write("Only four features are used in this simplified project. Real wine quality depends on many additional chemical, sensory and production factors. This prediction should not be treated as a professional quality certificate.")
    st.markdown("**Tools:** Python, Streamlit, Pandas, Scikit-learn, Matplotlib and Seaborn.")
