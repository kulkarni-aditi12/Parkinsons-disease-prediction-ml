import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Parkinson's Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}

.stApp {
    background:
        linear-gradient(rgba(240, 247, 252, 0.84), rgba(240, 247, 252, 0.86)),
        url('https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=1600&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

#MainMenu, footer, header {
    visibility: hidden !important;
}

[data-testid="stHeader"] {
    height: 0rem !important;
    background: transparent !important;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeaderActionElements"] {
    display: none !important;
}

[data-testid="stAppViewContainer"] > .main {
    padding-top: 0rem !important;
}

.block-container {
    padding-top: 22px !important;
    padding-bottom: 20px !important;
    max-width: 1150px !important;
}

/* remove accidental empty white boxes */
.element-container:empty {
    display: none !important;
}

.main-title {
    font-size: 2.4rem;
    font-weight: 800 !important;
    color: #17324d;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 1rem;
    color: #49627b;
    font-weight: 700 !important;
    margin-bottom: 16px;
}

.welcome-box {
    background: rgba(233, 245, 255, 0.98);
    border: 1.6px solid #d7eafb;
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 16px;
    color: #1d3a57;
    font-weight: 800 !important;
}

.section-box {
    background: rgba(255,255,255,0.96);
    border: 1.5px solid #d9e8f5;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 12px 30px rgba(22, 45, 72, 0.08);
    margin-bottom: 16px;
}

.section-heading {
    font-size: 1.2rem;
    font-weight: 800 !important;
    color: #1c3652;
    margin-bottom: 10px;
}

.result-low {
    background: linear-gradient(135deg, #e9fff1 0%, #dbf9e8 100%);
    border: 1.5px solid #c8edd8;
    color: #167c43;
    border-radius: 16px;
    padding: 14px 16px;
    font-size: 1.05rem;
    font-weight: 800 !important;
}

.result-moderate {
    background: linear-gradient(135deg, #fff8e6 0%, #ffefc8 100%);
    border: 1.5px solid #ffe1a0;
    color: #9b6700;
    border-radius: 16px;
    padding: 14px 16px;
    font-size: 1.05rem;
    font-weight: 800 !important;
}

.result-high {
    background: linear-gradient(135deg, #ffeaea 0%, #ffd7d7 100%);
    border: 1.5px solid #ffb7b7;
    color: #b3261e;
    border-radius: 16px;
    padding: 14px 16px;
    font-size: 1.05rem;
    font-weight: 800 !important;
}

.stSelectbox label, .stSlider label, .stTextInput label, .stMarkdown, p, span, div {
    font-weight: 700 !important;
}

.stTextInput div[data-baseweb="base-input"] {
    background: #f7fbff !important;
    border: 1.6px solid #d9e8f5 !important;
    border-radius: 14px !important;
}

.stTextInput input {
    color: #17324d !important;
    font-weight: 800 !important;
}

.stTextInput input::placeholder {
    color: #7b8794 !important;
    font-weight: 700 !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #f7fbff !important;
    border: 1.6px solid #d9e8f5 !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
    color: #17324d !important;
}

.stSlider {
    background: #f7fbff !important;
    border: 1.6px solid #d9e8f5 !important;
    border-radius: 14px !important;
    padding: 8px 12px 2px 12px !important;
}

.stButton > button {
    width: 100%;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.82rem 1rem !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    color: white !important;
    background: linear-gradient(90deg, #55c8e8 0%, #c23cf0 100%) !important;
    box-shadow: 0 10px 22px rgba(126, 90, 235, 0.24) !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
}

[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("parkinsons.data")

@st.cache_resource
def train_model():
    data = load_data()
    X = data.drop(columns=["name", "status"])
    y = data["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def convert_to_model_input(age, gender, voice, speech, tremor, movement, stiffness, balance):
    sample = {
        "MDVP:Fo(Hz)": 200.0,
        "MDVP:Fhi(Hz)": 215.0,
        "MDVP:Flo(Hz)": 190.0,
        "MDVP:Jitter(%)": 0.0022,
        "MDVP:Jitter(Abs)": 0.000010,
        "MDVP:RAP": 0.0011,
        "MDVP:PPQ": 0.0013,
        "Jitter:DDP": 0.0033,
        "MDVP:Shimmer": 0.011,
        "MDVP:Shimmer(dB)": 0.10,
        "Shimmer:APQ3": 0.0055,
        "Shimmer:APQ5": 0.0068,
        "MDVP:APQ": 0.0085,
        "Shimmer:DDA": 0.0165,
        "NHR": 0.0025,
        "HNR": 29.0,
        "RPDE": 0.40,
        "DFA": 0.72,
        "spread1": -7.1,
        "spread2": 0.17,
        "D2": 2.0,
        "PPE": 0.09,
    }

    reasons = []

    if age >= 60:
        sample["MDVP:Jitter(%)"] *= 1.08
        sample["MDVP:Shimmer"] *= 1.08
        sample["HNR"] -= 0.8
        reasons.append("higher age")

    if gender == "Female":
        sample["MDVP:Fo(Hz)"] = 220.0
        sample["MDVP:Fhi(Hz)"] = 235.0
        sample["MDVP:Flo(Hz)"] = 205.0

    if voice == "Slightly Shaky":
        sample["MDVP:Jitter(%)"] = 0.0060
        sample["MDVP:Jitter(Abs)"] = 0.000050
        sample["MDVP:RAP"] = 0.0030
        sample["MDVP:PPQ"] = 0.0035
        sample["Jitter:DDP"] = 0.0090
        sample["MDVP:Shimmer"] = 0.028
        sample["MDVP:Shimmer(dB)"] = 0.26
        sample["Shimmer:APQ3"] = 0.014
        sample["Shimmer:APQ5"] = 0.017
        sample["MDVP:APQ"] = 0.022
        sample["Shimmer:DDA"] = 0.042
        sample["NHR"] = 0.010
        sample["HNR"] = 22.0
        sample["RPDE"] = 0.52
        sample["PPE"] = 0.22
        reasons.append("slightly shaky voice")

    elif voice == "Very Shaky":
        sample["MDVP:Jitter(%)"] = 0.015
        sample["MDVP:Jitter(Abs)"] = 0.000120
        sample["MDVP:RAP"] = 0.0080
        sample["MDVP:PPQ"] = 0.0075
        sample["Jitter:DDP"] = 0.024
        sample["MDVP:Shimmer"] = 0.060
        sample["MDVP:Shimmer(dB)"] = 0.58
        sample["Shimmer:APQ3"] = 0.032
        sample["Shimmer:APQ5"] = 0.038
        sample["MDVP:APQ"] = 0.048
        sample["Shimmer:DDA"] = 0.096
        sample["NHR"] = 0.040
        sample["HNR"] = 15.0
        sample["RPDE"] = 0.60
        sample["DFA"] = 0.78
        sample["spread1"] = -5.2
        sample["spread2"] = 0.30
        sample["D2"] = 2.8
        sample["PPE"] = 0.33
        reasons.append("very shaky voice")

    if speech == "Slightly Unclear":
        sample["HNR"] -= 2.0
        sample["NHR"] += 0.004
        sample["RPDE"] += 0.03
        sample["PPE"] += 0.03
        reasons.append("slightly unclear speech")

    elif speech == "Unclear / Slurred":
        sample["HNR"] -= 5.0
        sample["NHR"] += 0.012
        sample["RPDE"] += 0.07
        sample["DFA"] += 0.03
        sample["spread1"] += 0.8
        sample["PPE"] += 0.08
        reasons.append("unclear or slurred speech")

    severity_score = 0

    if tremor == "Sometimes":
        severity_score += 1
        reasons.append("occasional tremor")
    elif tremor == "Frequent":
        severity_score += 2
        reasons.append("frequent tremor")

    if movement == "Slow":
        severity_score += 1
        reasons.append("slow movement")
    elif movement == "Very Slow":
        severity_score += 2
        reasons.append("very slow movement")

    if stiffness == "Mild":
        severity_score += 1
        reasons.append("mild stiffness")
    elif stiffness == "Severe":
        severity_score += 2
        reasons.append("severe stiffness")

    if balance == "Sometimes Unstable":
        severity_score += 1
        reasons.append("some balance difficulty")
    elif balance == "Frequently Unstable":
        severity_score += 2
        reasons.append("frequent balance difficulty")

    if severity_score == 1:
        sample["MDVP:Jitter(%)"] *= 1.05
        sample["MDVP:Shimmer"] *= 1.05
        sample["HNR"] -= 0.5
    elif severity_score == 2:
        sample["MDVP:Jitter(%)"] *= 1.10
        sample["MDVP:Shimmer"] *= 1.10
        sample["NHR"] += 0.001
        sample["HNR"] -= 1.0
        sample["PPE"] += 0.01
    elif severity_score == 3:
        sample["MDVP:Jitter(%)"] *= 1.18
        sample["MDVP:Shimmer"] *= 1.18
        sample["NHR"] += 0.002
        sample["HNR"] -= 1.5
        sample["RPDE"] += 0.02
        sample["PPE"] += 0.02
    elif severity_score >= 4:
        sample["MDVP:Jitter(%)"] *= 1.30
        sample["MDVP:Shimmer"] *= 1.30
        sample["NHR"] += 0.004
        sample["HNR"] -= 2.5
        sample["RPDE"] += 0.04
        sample["DFA"] += 0.02
        sample["spread1"] += 0.5
        sample["spread2"] += 0.02
        sample["D2"] += 0.2
        sample["PPE"] += 0.05

    sample["MDVP:Jitter(%)"] = max(0.0015, min(sample["MDVP:Jitter(%)"], 0.035))
    sample["MDVP:Shimmer"] = max(0.009, min(sample["MDVP:Shimmer"], 0.10))
    sample["NHR"] = max(0.0005, min(sample["NHR"], 0.35))
    sample["HNR"] = max(8.0, min(sample["HNR"], 35.0))
    sample["PPE"] = max(0.04, min(sample["PPE"], 0.45))

    return pd.DataFrame([sample]), reasons

model = train_model()

st.markdown('<div class="main-title">🧠 Parkinson’s Disease Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="welcome-box">Enter patient details below.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Basic Information</div>', unsafe_allow_html=True)
    patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
    age = st.slider("Age", 20, 90, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    voice = st.selectbox("Voice Condition", ["Normal", "Slightly Shaky", "Very Shaky"])
    speech = st.selectbox("Speech Clarity", ["Clear", "Slightly Unclear", "Unclear / Slurred"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Physical Symptoms</div>', unsafe_allow_html=True)
    tremor = st.selectbox("Hand Tremor", ["No", "Sometimes", "Frequent"])
    movement = st.selectbox("Movement Speed", ["Normal", "Slow", "Very Slow"])
    stiffness = st.selectbox("Muscle Stiffness", ["No", "Mild", "Severe"])
    balance = st.selectbox("Balance While Walking", ["Normal", "Sometimes Unstable", "Frequently Unstable"])
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("Predict Risk"):
    input_df, reasons = convert_to_model_input(age, gender, voice, speech, tremor, movement, stiffness, balance)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    risk_score = probability[1] * 100

    st.markdown('<div class="section-heading">Prediction Result</div>', unsafe_allow_html=True)

    if risk_score < 40:
        st.markdown(f'<div class="result-low">Low Risk ({risk_score:.2f}%)</div>', unsafe_allow_html=True)
    elif risk_score < 70:
        st.markdown(f'<div class="result-moderate">Moderate Risk ({risk_score:.2f}%)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-high">High Risk ({risk_score:.2f}%)</div>', unsafe_allow_html=True)

    if patient_name.strip():
        st.write(f"**Patient Name:** {patient_name}")

    st.write(
        f"**Model prediction:** {'Parkinson-like pattern detected' if prediction == 1 else 'Healthy-like pattern detected'}"
    )

    if reasons:
        st.write("**Main contributing factors:** " + ", ".join(reasons))

    st.subheader("Suggestions")

    if risk_score < 40:
        st.success("""
Maintain a healthy lifestyle  
Continue regular physical activity  
Sleep properly and manage stress  
Do periodic health checkups if symptoms appear  
""")
    elif risk_score < 70:
        st.warning("""
Consult a doctor for early screening  
Monitor symptoms regularly  
Do light exercise like walking or stretching  
Maintain healthy food and sleep habits  
Avoid delaying medical advice if symptoms increase  
""")
    else:
        st.error("""
Consult a neurologist as soon as possible  
Go for proper clinical and neurological evaluation   
Take family support and monitor movement, speech, and tremors carefully  
""")

    with st.expander("Show values used for prediction"):
        preview_df = input_df.T
        preview_df.columns = ["Value"]
        st.dataframe(preview_df, use_container_width=True)