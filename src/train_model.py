import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from config import PROCESSED_DATA_PATH, MODEL_PATH

def train_and_evaluate():
    # Load the processed data
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"🧠 Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()

