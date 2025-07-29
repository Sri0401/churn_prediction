import pandas as pd
import joblib
import os
import argparse

from config import MODEL_PATH, PROCESSED_DATA_PATH  # reuse data path if needed

def load_model():
    model = joblib.load(MODEL_PATH)
    print("🧠 Model loaded successfully.")
    return model

def run_inference(model, data):
    predictions = model.predict(data)
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Run churn prediction inference.")
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to trained model file')
    parser.add_argument('--input', type=str, default=PROCESSED_DATA_PATH, help='Path to input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Path to output predictions CSV file')
    args = parser.parse_args()

    # Load model
    model = joblib.load(args.model)
    print(f"🧠 Model loaded from {args.model}")

    # Load data for inference
    df = pd.read_csv(args.input)

    # Drop target column
    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)

    # Run inference
    preds = run_inference(model, df)

    # Attach predictions
    df["Predicted_Churn"] = preds

    # Save predictions
    output_path = args.output or os.path.join(os.path.dirname(args.input), "predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"📁 Predictions saved at {output_path}")

if __name__ == "__main__":
    main()
