import pandas as pd

desc_df = pd.read_csv("data/disease_description.csv")
prec_df = pd.read_csv("data/disease_precautions.csv")

def get_description(disease):
    row = desc_df[desc_df["Disease"].str.lower() == disease.lower()]
    return row["Description"].values[0] if not row.empty else "Description not found."

def get_precautions(disease):
    row = prec_df[prec_df["Disease"].str.lower() == disease.lower()]
    if not row.empty:
        return [row[f"Precaution_{i}"].values[0] for i in range(1, 5)]
    return ["No precautions found."]
