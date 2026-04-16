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
        n_estimators=400,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


@st.cache_data
def get_reference_profiles():
    data = load_data()
    feature_cols = [col for col in data.columns if col not in ["name", "status"]]

    healthy = data[data["status"] == 0][feature_cols]
    parkinsons = data[data["status"] == 1][feature_cols]

    healthy_median = healthy.median()
    parkinsons_median = parkinsons.median()
    q05 = data[feature_cols].quantile(0.05)
    q95 = data[feature_cols].quantile(0.95)

    return feature_cols, healthy_median, parkinsons_median, q05, q95


def convert_to_model_input(age, gender, voice, speech, tremor, movement, stiffness, balance):
    feature_cols, healthy_med, pd_med, q05, q95 = get_reference_profiles()

    reasons = []

    # Build symptom score in a controlled way
    symptom_score = 0.0

    age_score = 0.0
    if age >= 70:
        age_score = 1.5
        reasons.append("higher age")
    elif age >= 60:
        age_score = 1.0
        reasons.append("higher age")
    elif age >= 50:
        age_score = 0.5

    voice_map = {
        "Normal": 0.0,
        "Slightly Shaky": 2.0,
        "Very Shaky": 3.6
    }
    speech_map = {
        "Clear": 0.0,
        "Slightly Unclear": 1.5,
        "Unclear / Slurred": 3.0
    }
    tremor_map = {
        "No": 0.0,
        "Sometimes": 1.2,
        "Frequent": 2.4
    }
    movement_map = {
        "Normal": 0.0,
        "Slow": 1.2,
        "Very Slow": 2.2
    }
    stiffness_map = {
        "No": 0.0,
        "Mild": 1.0,
        "Severe": 2.0
    }
    balance_map = {
        "Normal": 0.0,
        "Sometimes Unstable": 1.1,
        "Frequently Unstable": 2.1
    }

    symptom_score += age_score
    symptom_score += voice_map[voice]
    symptom_score += speech_map[speech]
    symptom_score += tremor_map[tremor]
    symptom_score += movement_map[movement]
    symptom_score += stiffness_map[stiffness]
    symptom_score += balance_map[balance]

    if voice == "Slightly Shaky":
        reasons.append("slightly shaky voice")
    elif voice == "Very Shaky":
        reasons.append("very shaky voice")

    if speech == "Slightly Unclear":
        reasons.append("slightly unclear speech")
    elif speech == "Unclear / Slurred":
        reasons.append("unclear or slurred speech")

    if tremor == "Sometimes":
        reasons.append("occasional tremor")
    elif tremor == "Frequent":
        reasons.append("frequent tremor")

    if movement == "Slow":
        reasons.append("slow movement")
    elif movement == "Very Slow":
        reasons.append("very slow movement")

    if stiffness == "Mild":
        reasons.append("mild stiffness")
    elif stiffness == "Severe":
        reasons.append("severe stiffness")

    if balance == "Sometimes Unstable":
        reasons.append("some balance difficulty")
    elif balance == "Frequently Unstable":
        reasons.append("frequent balance difficulty")

    # Normalize symptom score
    max_score = 16.8
    severity_ratio = min(symptom_score / max_score, 1.0)

    # Start from healthy median and move gradually toward Parkinson median
    sample = healthy_med + (pd_med - healthy_med) * severity_ratio

    # Controlled age effect
    if age >= 60:
        sample["MDVP:Jitter(%)"] *= 1.04
        sample["MDVP:Shimmer"] *= 1.04
        sample["NHR"] *= 1.03
        sample["HNR"] *= 0.98
    if age >= 70:
        sample["MDVP:Jitter(%)"] *= 1.03
        sample["MDVP:Shimmer"] *= 1.03
        sample["PPE"] *= 1.04

    # Controlled gender effect only on pitch-related features
    if gender == "Female":
        sample["MDVP:Fo(Hz)"] *= 1.08
        sample["MDVP:Fhi(Hz)"] *= 1.06
        sample["MDVP:Flo(Hz)"] *= 1.06

    # Voice effect
    if voice == "Slightly Shaky":
        sample["MDVP:Jitter(%)"] *= 1.12
        sample["MDVP:Jitter(Abs)"] *= 1.12
        sample["MDVP:RAP"] *= 1.10
        sample["MDVP:PPQ"] *= 1.10
        sample["Jitter:DDP"] *= 1.10
        sample["MDVP:Shimmer"] *= 1.10
        sample["MDVP:Shimmer(dB)"] *= 1.08
        sample["Shimmer:APQ3"] *= 1.08
        sample["Shimmer:APQ5"] *= 1.08
        sample["MDVP:APQ"] *= 1.08
        sample["Shimmer:DDA"] *= 1.08
        sample["NHR"] *= 1.06
        sample["HNR"] *= 0.96
        sample["PPE"] *= 1.08

    elif voice == "Very Shaky":
        sample["MDVP:Jitter(%)"] *= 1.25
        sample["MDVP:Jitter(Abs)"] *= 1.20
        sample["MDVP:RAP"] *= 1.18
        sample["MDVP:PPQ"] *= 1.18
        sample["Jitter:DDP"] *= 1.18
        sample["MDVP:Shimmer"] *= 1.22
        sample["MDVP:Shimmer(dB)"] *= 1.18
        sample["Shimmer:APQ3"] *= 1.16
        sample["Shimmer:APQ5"] *= 1.16
        sample["MDVP:APQ"] *= 1.16
        sample["Shimmer:DDA"] *= 1.16
        sample["NHR"] *= 1.12
        sample["HNR"] *= 0.90
        sample["RPDE"] *= 1.08
        sample["PPE"] *= 1.18

    # Speech effect
    if speech == "Slightly Unclear":
        sample["HNR"] *= 0.96
        sample["NHR"] *= 1.05
        sample["RPDE"] *= 1.04
        sample["PPE"] *= 1.05

    elif speech == "Unclear / Slurred":
        sample["HNR"] *= 0.90
        sample["NHR"] *= 1.10
        sample["RPDE"] *= 1.08
        sample["DFA"] *= 1.03
        sample["spread1"] *= 0.93 if sample["spread1"] < 0 else 1.07
        sample["PPE"] *= 1.10

    # Motor severity effect
    motor_score = tremor_map[tremor] + movement_map[movement] + stiffness_map[stiffness] + balance_map[balance]

    if motor_score >= 1.5:
        sample["MDVP:Jitter(%)"] *= 1.03
        sample["MDVP:Shimmer"] *= 1.03
        sample["HNR"] *= 0.99

    if motor_score >= 3.5:
        sample["MDVP:Jitter(%)"] *= 1.05
        sample["MDVP:Shimmer"] *= 1.05
        sample["NHR"] *= 1.04
        sample["RPDE"] *= 1.03
        sample["PPE"] *= 1.04

    if motor_score >= 5.5:
        sample["MDVP:Jitter(%)"] *= 1.08
        sample["MDVP:Shimmer"] *= 1.08
        sample["NHR"] *= 1.06
        sample["HNR"] *= 0.95
        sample["RPDE"] *= 1.05
        sample["DFA"] *= 1.02
        sample["PPE"] *= 1.07

    # Keep values realistic using dataset bounds
    sample = sample.clip(lower=q05, upper=q95)

    # Make sure order matches training columns
    input_df = pd.DataFrame([sample])[feature_cols]

    return input_df, reasons, symptom_score


def compute_final_risk(model_prob, symptom_score, voice, speech, tremor, movement, stiffness, balance):
    # Convert symptom score to 0-100 scale in a controlled way
    symptom_risk = min((symptom_score / 16.8) * 100, 100)

    # Blend model output with symptom score
    final_risk = 0.72 * model_prob + 0.28 * symptom_risk

    # Make clearly healthy cases stay low
    if (
        voice == "Normal"
        and speech == "Clear"
        and tremor == "No"
        and movement == "Normal"
        and stiffness == "No"
        and balance == "Normal"
    ):
        final_risk = min(final_risk, 24)

    # Gentle moderate shaping
    mild_symptoms_count = sum([
        voice == "Slightly Shaky",
        speech == "Slightly Unclear",
        tremor == "Sometimes",
        movement == "Slow",
        stiffness == "Mild",
        balance == "Sometimes Unstable"
    ])

    severe_symptoms_count = sum([
        voice == "Very Shaky",
        speech == "Unclear / Slurred",
        tremor == "Frequent",
        movement == "Very Slow",
        stiffness == "Severe",
        balance == "Frequently Unstable"
    ])

    if severe_symptoms_count == 0 and mild_symptoms_count >= 2:
        final_risk = min(max(final_risk, 42), 68)

    if severe_symptoms_count >= 3:
        final_risk = max(final_risk, 72)

    return round(float(max(0, min(final_risk, 100))), 2)


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
    input_df, reasons, symptom_score = convert_to_model_input(
        age, gender, voice, speech, tremor, movement, stiffness, balance
    )

    probability = model.predict_proba(input_df)[0]
    model_risk = probability[1] * 100

    risk_score = compute_final_risk(
        model_risk, symptom_score,
        voice, speech, tremor, movement, stiffness, balance
    )

    prediction = 1 if risk_score >= 50 else 0

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
        st.dataframe(preview_df, width="stretch")