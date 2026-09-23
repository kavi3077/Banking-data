import pandas as pd
import numpy as np

df = pd.read_csv("banking_dataset_cleaned.csv")

print("ORIGINAL DATASET SHAPE")
print(df.shape)

print("\nORIGINAL DATASET")
print(df.head())

df["Transaction_Amount_Category"] = np.select(
    [
        df["Transaction_Amount"] < 5000,
        df["Transaction_Amount"].between(5000, 20000)
    ],
    [
        "Low",
        "Medium"
    ],
    default="High"
)

df["Account_Balance_Status"] = np.select(
    [
        df["Account_Balance"] < 10000,
        df["Account_Balance"].between(10000, 50000)
    ],
    [
        "Low",
        "Medium"
    ],
    default="High"
)

df["Transaction_Indicator"] = np.where(
    df["Transaction_Type"].str.lower().isin(
        ["deposit", "credit"]
    ),
    "Credit",
    "Debit"
)

df["High_Value_Transaction_Indicator"] = np.where(
    df["Transaction_Amount"] > 20000,
    "Yes",
    "No"
)

df["Transaction_to_Balance_Ratio"] = np.where(
    df["Account_Balance"] > 0,
    df["Transaction_Amount"] /
    df["Account_Balance"].replace(0, np.nan),
    np.nan
)

df["Transaction_to_Balance_Ratio"] = (
    df["Transaction_to_Balance_Ratio"].round(2)
)

print("\nTRANSFORMED DATASET")
print(df.head())

print("\nTRANSACTION AMOUNT CATEGORY")
print(df["Transaction_Amount_Category"].value_counts())

print("\nACCOUNT BALANCE STATUS")
print(df["Account_Balance_Status"].value_counts())

print("\nTRANSACTION INDICATOR")
print(df["Transaction_Indicator"].value_counts())

print("\nHIGH VALUE TRANSACTION INDICATOR")
print(
    df["High_Value_Transaction_Indicator"].value_counts()
)

print("\nDATA VALIDATION")

print("Invalid Transaction Categories:",
      (~df["Transaction_Amount_Category"].isin(
          ["Low", "Medium", "High"]
      )).sum())

print("Invalid Balance Status:",
      (~df["Account_Balance_Status"].isin(
          ["Low", "Medium", "High"]
      )).sum())

print("Invalid Transaction Indicators:",
      (~df["Transaction_Indicator"].isin(
          ["Credit", "Debit"]
      )).sum())

print("Invalid High Value Indicators:",
      (~df["High_Value_Transaction_Indicator"].isin(
          ["Yes", "No"]
      )).sum())

print("Invalid Transaction Ratios:",
      np.isinf(
          df["Transaction_to_Balance_Ratio"]
      ).sum())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nFINAL DATASET SHAPE")
print(df.shape)

df.to_csv(
    "banking_dataset_transformed.csv",
    index=False
)

print("\nTRANSFORMED DATASET SAVED SUCCESSFULLY")