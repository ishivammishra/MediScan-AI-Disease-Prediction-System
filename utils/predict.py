import pickle
import pandas as pd

# Load trained model and encoders
with open("models/trained_model.pkl", "rb") as f:
    model, mlb, le = pickle.load(f)

def predict_disease(symptoms: list):
    input_vector = mlb.transform([symptoms])
    pred = model.predict(input_vector)[0]
    prob = model.predict_proba(input_vector)[0].max() * 100
    disease = le.inverse_transform([pred])[0]
    return disease, prob
