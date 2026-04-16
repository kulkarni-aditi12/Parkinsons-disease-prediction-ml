# Parkinson’s Disease Prediction using Machine Learning

This project is a Streamlit-based web application that predicts the risk of Parkinson’s disease using a machine learning model trained on biomedical voice measurements.

---

## Overview

The application uses a Random Forest classifier trained on a dataset containing voice-related features to determine whether a patient shows Parkinson-like patterns.

User inputs such as age, gender, voice condition, speech clarity, tremor, movement, stiffness, and balance are converted into feature values and passed to the model for prediction.

---

## Tech Stack

* Python
* Pandas
* Scikit-learn
* Streamlit

---

## Features

* Interactive user interface built with Streamlit
* Random Forest model for prediction
* Risk classification based on prediction probability (Low, Moderate, High)
* Accepts patient details and symptom-based inputs
* Displays prediction result and contributing factors
* Provides basic suggestions based on predicted risk

---

## Methodology

1. Load dataset using Pandas
2. Separate input features and target variable
3. Split dataset into training and testing sets
4. Train a Random Forest classifier
5. Convert user inputs into model-compatible feature values
6. Predict disease status and probability
7. Display result in the interface

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas streamlit scikit-learn
```

### 2. Run the application

```bash
streamlit run app.py
```


## Project Structure

```
parkinsons-disease-prediction-ml/
│
├── app.py
├── parkinsons.data
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

The dataset contains biomedical voice measurements used for Parkinson’s disease detection.
It is loaded from the file `parkinsons.data`.

