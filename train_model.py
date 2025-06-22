import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

# Load symptom-disease dataset
df = pd.read_csv("data/symptoms_by_disease.csv")
df.fillna("", inplace=True)

# Combine all symptom columns into a list
symptom_cols = [f"Symptom_{i}" for i in range(1, 18)]
df["symptoms"] = df[symptom_cols].values.tolist()
df["symptoms"] = df["symptoms"].apply(lambda x: [i.strip() for i in x if i.strip() != ""])

# Encode symptoms
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptoms"])

# Encode diseases
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model, encoders
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump((model, mlb, le), f)
