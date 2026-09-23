import pandas as pd
import numpy as np
df = pd.read_csv("banking_dataset_150.csv")
print("ORIGINAL DATASET SHAPE")
print(df.shape)
print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())
print("\nDUPLICATES BEFORE CLEANING")
print(df.duplicated().sum())
df = df.drop_duplicates().copy()
text_columns = [
    "Customer_ID",
    "Customer_Name",
    "Gender",
    "Account_Number",
    "Account_Type",
    "Branch",
    "Transaction_Type"
]
for col in text_columns:
    df[col] = df[col].astype("string").str.strip()
df["Customer_ID"] = df["Customer_ID"].str.upper()
df["Customer_Name"] = df["Customer_Name"].str.title()
df["Gender"] = df["Gender"].str.title()
df["Account_Number"] = df["Account_Number"].str.upper()
df["Account_Type"] = df["Account_Type"].str.title()
df["Branch"] = df["Branch"].str.title()
df["Transaction_Type"] = df["Transaction_Type"].str.title()
df["Branch"] = df["Branch"].fillna(df["Branch"].mode()[0])
numeric_columns = [
    "Age",
    "Account_Balance",
    "Transaction_Amount"
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["Age"] = df["Age"].replace(
    [np.inf, -np.inf], np.nan
)
df.loc[
    (df["Age"] < 18) | (df["Age"] > 100),
    "Age"
] = np.nan
df["Age"] = df["Age"].fillna(
    df["Age"].median()
).round().astype("int64")
df["Account_Balance"] = df["Account_Balance"].replace(
    [np.inf, -np.inf], np.nan
)
df.loc[
    df["Account_Balance"] < 0,
    "Account_Balance"
] = np.nan
df["Account_Balance"] = df["Account_Balance"].fillna(
    df["Account_Balance"].median()
).astype("float64")
df["Transaction_Amount"] = df["Transaction_Amount"].replace(
    [np.inf, -np.inf], np.nan
)
df.loc[
    df["Transaction_Amount"] <= 0,
    "Transaction_Amount"
] = np.nan
df["Transaction_Amount"] = df["Transaction_Amount"].fillna(
    df["Transaction_Amount"].median()
).astype("float64")
df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    errors="coerce"
)
print("\nCLEANED DATASET")
print(df.head())
print("\nCLEANED DATASET SHAPE")
print(df.shape)
print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())
print("\nDUPLICATES AFTER CLEANING")
print(df.duplicated().sum())
print("\nDATA TYPES AFTER CLEANING")
print(df.dtypes)
print("\nDATA VALIDATION")
print("Invalid Ages:", (
    ~df["Age"].between(18, 100)
).sum())
print("Invalid Account Balances:", (
    df["Account_Balance"] < 0
).sum())
print("Invalid Transaction Amounts:", (
    df["Transaction_Amount"] <= 0
).sum())
print("Invalid Transaction Dates:",
      df["Transaction_Date"].isna().sum())
print("Duplicate Customer IDs:",
      df["Customer_ID"].duplicated().sum())
print("Duplicate Account Numbers:",
      df["Account_Number"].duplicated().sum())
df.to_csv("banking_dataset_cleaned.csv", index=False)
print("\nCLEANED DATASET SAVED SUCCESSFULLY")