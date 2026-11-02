# WineWise — Wine Quality Classification Website

A beginner-friendly Python + Streamlit + Machine Learning project. It predicts whether a red wine sample belongs to the **Low**, **Medium**, or **High** quality category using four chemical features and a Random Forest classifier.

## Features
- Home dashboard and project overview
- Interactive prediction form
- Dataset explorer with category chart and alcohol histogram
- Model performance page with accuracy, confusion matrix, classification report and feature importance
- Optional CSV upload
- Automatic model training when the app starts

## 1. Install Python and VS Code
1. Install Python 3.10 or newer from https://www.python.org/downloads/
2. During installation on Windows, tick **Add Python to PATH**.
3. Install Visual Studio Code from https://code.visualstudio.com/
4. In VS Code, install the **Python** extension by Microsoft.

## 2. Open the project
1. Extract `Wine_Quality_Classification_Project.zip`.
2. Open VS Code.
3. Select **File → Open Folder** and choose the extracted `wine_quality_classifier_project` folder.
4. Open **Terminal → New Terminal**.

## 3. Create a virtual environment

### Windows PowerShell
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install libraries
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the website
```bash
streamlit run app.py
```
A browser tab should open automatically. If it does not, copy the local URL shown in the terminal (usually `http://localhost:8501`) into your browser.

The first run downloads the red wine quality CSV dataset from the UCI Machine Learning Repository. An internet connection is needed for this first download. Later runs use the saved CSV file.

## 6. Optional: train and save a separate model
The website trains its model automatically. If you also want a standalone saved model file, run:
```bash
python train_model.py
```
This creates `wine_quality_model.pkl` and `model_metrics.json`.

## Dataset
The project uses the UCI Wine Quality red wine dataset. The original dataset contains a numeric `quality` score and chemical measurements. This project groups the score as:
- Low: 4 or below
- Medium: 5–6
- High: 7 or above

Dataset source: https://archive.ics.uci.edu/dataset/186/wine+quality

You can also download the red wine CSV from the UCI repository or use the Kaggle Wine Quality dataset. If uploading a CSV in the app, it must contain these columns with their standard names:
`alcohol`, `volatile acidity`, `pH`, `sulphates`, `quality`.

## How to explain the project to your teacher
1. **Problem:** Wine quality is related to chemical properties. We want to predict a quality category from those measurements.
2. **Dataset:** We use labelled red wine records. Each record has chemical measurements and a quality score.
3. **Preprocessing:** We select four input columns, remove incomplete records, and convert numeric scores into three categories.
4. **Training:** We split the data into training and testing sets. A Random Forest classifier learns patterns from the training set.
5. **Prediction:** The website takes four values from the user and sends them to the trained model.
6. **Evaluation:** We show test accuracy, a confusion matrix, precision, recall and F1-score.
7. **Limitations:** This is a learning demonstration, not a professional wine-testing tool. Only four features are used.

## Project structure
```text
wine_quality_classifier_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Troubleshooting
- **`python` is not recognized:** reinstall Python and select "Add Python to PATH", then restart VS Code.
- **`streamlit` is not recognized:** activate `.venv` and run `python -m pip install -r requirements.txt`.
- **PowerShell activation error:** use the Command Prompt activation command above.
- **Dataset download error:** check internet access, or download the UCI red wine CSV manually and place it in this project folder as `winequality-red.csv`.
- **CSV upload error:** confirm the file has the five required column names listed above.
