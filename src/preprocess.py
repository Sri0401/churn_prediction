import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def preprocess_telco_data():
    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # Drop unnecessary column
    df.drop('customerID', axis=1, inplace=True)

    # Handle missing/blank TotalCharges (convert to numeric)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Identify categorical and numerical features
    categorical_cols = df.select_dtypes(include='object').columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).drop('Churn', axis=1).columns

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save the processed data
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Preprocessing complete. Saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_telco_data()
