from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import os
from src.config import MODEL_PATH

app = FastAPI()

model = joblib.load(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)
    preds = model.predict(df)
    df["Predicted_Churn"] = preds
    output_path = os.path.join(os.path.dirname(MODEL_PATH), "api_predictions.csv")
    df.to_csv(output_path, index=False)
    return {"message": f"Predictions saved at {output_path}", "predictions": preds.tolist()}
