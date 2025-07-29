from src.data_load import load_data

if __name__ == "__main__":
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(df.head())