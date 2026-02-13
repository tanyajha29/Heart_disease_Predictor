# HeartGuard - AI-Powered Heart Disease Risk Assessment

HeartGuard is a Streamlit web application that predicts the likelihood of heart
disease based on patient health metrics. It uses a Random Forest model trained
on the Cleveland Heart Disease dataset (UCI) and returns a risk label with a
confidence score, plus a downloadable PDF report.

Demo video: `sample_vdeo.mp4`

**Features**
- Interactive assessment form covering key clinical variables
- Risk prediction with confidence score
- Results page with summary metrics and recommendations
- PDF report export for sharing
- Clean, responsive Streamlit UI

**Tech Stack**
- Python
- Streamlit
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- reportlab (PDF export)

**Dataset**
- Source: Cleveland Heart Disease dataset (UCI)
- Local file: `data/heart.csv`
- The training script assigns column names and converts the `target` field to a
  binary label (0 = no disease, 1 = disease).

**Model**
- StandardScaler for feature normalization
- RandomForestClassifier (`n_estimators=100`, `max_depth=5`)
- Training script: `utils/processing.py`
- Optional evaluation: `evaluate_model.py` (accuracy, classification report,
  confusion matrix)

**Getting Started**
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Train the model:
```bash
python utils/processing.py
```
This creates `model/heart_model.pkl` and `model/scaler.pkl`.
3. Run the app:
```bash
streamlit run app.py
```

**Project Structure**
```text
Heart_Disease_Predictor/
├─ app.py
├─ evaluate_model.py
├─ requirements.txt
├─ sample_vdeo.mp4
├─ data/
│  └─ heart.csv
├─ model/
│  ├─ heart_model.pkl
│  └─ scaler.pkl
└─ utils/
   └─ processing.py
```

**Notes**
- For educational use only; not a substitute for professional medical advice.
- If you update the dataset or model parameters, re-run `utils/processing.py`.
