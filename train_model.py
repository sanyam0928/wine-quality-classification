"""Optional script: train and save the WineWise model locally."""
import urllib.request
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import json

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
CSV = "winequality-red.csv"
FEATURES = ["alcohol", "volatile acidity", "pH", "sulphates"]

try:
    df = pd.read_csv(CSV, sep=";")
except FileNotFoundError:
    print("Downloading the UCI red wine quality dataset...")
    urllib.request.urlretrieve(URL, CSV)
    df = pd.read_csv(CSV, sep=";")

df["category"] = pd.cut(df["quality"], [-float("inf"), 4, 6, float("inf")],
                        labels=["Low", "Medium", "High"])
df = df.dropna(subset=FEATURES + ["category"])
X, y = df[FEATURES], df["category"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced",
        min_samples_split=4
    ))
])
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
joblib.dump(model, "wine_quality_model.pkl")
with open("model_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"accuracy": accuracy,
               "classification_report": classification_report(y_test, preds, output_dict=True)},
              f, indent=2)
print(f"Model saved. Test accuracy: {accuracy:.2%}")
