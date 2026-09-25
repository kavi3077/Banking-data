import pandas as pd
import numpy as np
df = pd.read_csv("banking_dataset.csv")
print("\nDAY 1")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nFirst 5 Rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())


df = df.drop_duplicates().copy()
text_columns = [
    "Customer_ID", "Customer_Name", "Gender", "Account_Number",
    "Account_Type", "Branch", "Transaction_ID",
    "Transaction_Type", "Loan_Type", "Loan_Status"
]
for col in text_columns:
    df[col] = df[col].astype("string").str.strip()
df["Customer_ID"] = df["Customer_ID"].str.upper()
df["Customer_Name"] = df["Customer_Name"].str.title()
df["Gender"] = df["Gender"].str.title()
df["Account_Number"] = df["Account_Number"].str.upper()
df["Account_Type"] = df["Account_Type"].str.title()
df["Branch"] = df["Branch"].str.title()
df["Transaction_ID"] = df["Transaction_ID"].str.upper()
df["Transaction_Type"] = df["Transaction_Type"].str.title()
df["Loan_Type"] = df["Loan_Type"].str.title()
df["Loan_Status"] = df["Loan_Status"].str.title()
numeric_columns = ["Age", "Transaction_Amount", "Account_Balance", "Loan_Amount"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
df.loc[(df["Age"] < 18) | (df["Age"] > 100), "Age"] = np.nan
df.loc[df["Transaction_Amount"] < 0, "Transaction_Amount"] = np.nan
df.loc[df["Account_Balance"] < 0, "Account_Balance"] = np.nan
df.loc[df["Loan_Amount"] < 0, "Loan_Amount"] = np.nan
df["Gender"] = df["Gender"].replace({"Unknown": pd.NA})
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Branch"] = df["Branch"].fillna("Unknown")
df["Loan_Type"] = df["Loan_Type"].fillna("No Loan")
df["Loan_Status"] = df["Loan_Status"].fillna("No Loan")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Transaction_Amount"] = df["Transaction_Amount"].fillna(df["Transaction_Amount"].median())
df["Account_Balance"] = df["Account_Balance"].fillna(df["Account_Balance"].median())
df["Loan_Amount"] = df["Loan_Amount"].fillna(0)
df.loc[df["Loan_Type"].eq("No Loan"), "Loan_Amount"] = 0
df.loc[df["Loan_Type"].eq("No Loan"), "Loan_Status"] = "No Loan"

print("\nDAY 2")
print("Shape After Removing Duplicates:", df.shape)
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
print("\nDuplicates After Cleaning:", df.duplicated().sum())
print("\nCleaned Data Types:")
print(df.dtypes)
df.to_csv("banking_dataset_cleaned.csv", index=False)


df["Transaction_Amount_Category"] = np.select(
    [
        df["Transaction_Amount"] < 5000,
        df["Transaction_Amount"].between(5000, 20000)
    ],
    ["Low", "Medium"],
    default="High"
)
df["Account_Balance_Status"] = np.select(
    [
        df["Account_Balance"] < 10000,
        df["Account_Balance"].between(10000, 50000)
    ],
    ["Low", "Medium"],
    default="High"
)
df["Transaction_Indicator"] = np.where(
    df["Transaction_Type"].str.lower().isin(["deposit", "credit"]),
    "Credit",
    "Debit"
)
df["High_Value_Transaction_Indicator"] = np.where(
    df["Transaction_Amount"] > 20000,
    "Yes",
    "No"
)
df["Loan_Indicator"] = np.where(
    (df["Loan_Type"] != "No Loan") & (df["Loan_Amount"] > 0),
    "Yes",
    "No"
)
df["High_Loan_Indicator"] = np.where(
    df["Loan_Amount"] > 500000,
    "Yes",
    "No"
)

print("\nDAY 3")
print(df[
    [
        "Transaction_Amount",
        "Transaction_Amount_Category",
        "Account_Balance",
        "Account_Balance_Status",
        "Transaction_Indicator",
        "High_Value_Transaction_Indicator",
        "Loan_Indicator",
        "High_Loan_Indicator"
    ]
].head(10))
print("\nTransaction Amount Categories:")
print(df["Transaction_Amount_Category"].value_counts())
print("\nAccount Balance Status:")
print(df["Account_Balance_Status"].value_counts())
df.to_csv("banking_dataset_transformed.csv", index=False)


print("\nDAY 4")
total_transaction_amount = df["Transaction_Amount"].sum()
average_transaction_amount = df["Transaction_Amount"].mean()
maximum_transaction = df["Transaction_Amount"].max()
minimum_transaction = df["Transaction_Amount"].min()
print("\nTRANSACTION SUMMARY")
print("Total Transaction Amount:", round(total_transaction_amount, 2))
print("Average Transaction Amount:", round(average_transaction_amount, 2))
print("Maximum Transaction:", round(maximum_transaction, 2))
print("Minimum Transaction:", round(minimum_transaction, 2))
transactions_by_type = df.groupby("Transaction_Type").agg(
    Transaction_Count=("Transaction_ID", "count"),
    Total_Transaction_Amount=("Transaction_Amount", "sum"),
    Average_Transaction_Amount=("Transaction_Amount", "mean")
).sort_values("Total_Transaction_Amount", ascending=False)
print("\nTRANSACTIONS BY TYPE")
print(transactions_by_type)
transactions_by_branch = df.groupby("Branch").agg(
    Transaction_Count=("Transaction_ID", "count"),
    Total_Transaction_Amount=("Transaction_Amount", "sum"),
    Average_Transaction_Amount=("Transaction_Amount", "mean"),
    Average_Account_Balance=("Account_Balance", "mean")
).sort_values("Total_Transaction_Amount", ascending=False)
print("\nTRANSACTIONS BY BRANCH")
print(transactions_by_branch)
customer_activity = df.groupby(["Customer_ID", "Customer_Name"]).agg(
    Transaction_Count=("Transaction_ID", "count"),
    Total_Transaction_Amount=("Transaction_Amount", "sum"),
    Average_Transaction_Amount=("Transaction_Amount", "mean"),
    Account_Balance=("Account_Balance", "mean")
).sort_values("Total_Transaction_Amount", ascending=False)
print("\nTOP 10 ACTIVE CUSTOMERS")
print(customer_activity.head(10))
account_analysis = df.groupby("Account_Type").agg(
    Number_of_Accounts=("Account_Number", "nunique"),
    Average_Balance=("Account_Balance", "mean"),
    Minimum_Balance=("Account_Balance", "min"),
    Maximum_Balance=("Account_Balance", "max")
).sort_values("Average_Balance", ascending=False)
print("\nACCOUNT TYPE ANALYSIS")
print(account_analysis)
loan_customers = df[df["Loan_Indicator"] == "Yes"]
loan_analysis = loan_customers.groupby("Loan_Type").agg(
    Number_of_Loans=("Customer_ID", "count"),
    Total_Loan_Amount=("Loan_Amount", "sum"),
    Average_Loan_Amount=("Loan_Amount", "mean")
).sort_values("Total_Loan_Amount", ascending=False)
print("\nLOAN ANALYSIS")
print(loan_analysis)
loan_status_analysis = loan_customers.groupby("Loan_Status").agg(
    Number_of_Loans=("Customer_ID", "count"),
    Total_Loan_Amount=("Loan_Amount", "sum")
).sort_values("Total_Loan_Amount", ascending=False)
print("\nLOAN STATUS ANALYSIS")
print(loan_status_analysis)
high_value_transactions = df[df["High_Value_Transaction_Indicator"] == "Yes"]
print("\nHIGH VALUE TRANSACTIONS")
print("Count:", len(high_value_transactions))
print("Total Amount:", round(high_value_transactions["Transaction_Amount"].sum(), 2))
category_analysis = df.groupby("Transaction_Amount_Category").agg(
    Transaction_Count=("Transaction_ID", "count"),
    Total_Amount=("Transaction_Amount", "sum"),
    Average_Amount=("Transaction_Amount", "mean")
).sort_values("Total_Amount", ascending=False)
print("\nTRANSACTION CATEGORY ANALYSIS")
print(category_analysis)
balance_status_analysis = df.groupby("Account_Balance_Status").agg(
    Customer_Count=("Customer_ID", "count"),
    Average_Balance=("Account_Balance", "mean")
).sort_values("Average_Balance", ascending=False)
print("\nACCOUNT BALANCE STATUS ANALYSIS")
print(balance_status_analysis)
print("\nOVERALL BANKING METRICS")
print("Unique Customers:", df["Customer_ID"].nunique())
print("Unique Accounts:", df["Account_Number"].nunique())
print("Total Transactions:", df["Transaction_ID"].count())
print("Total Transaction Amount:", round(df["Transaction_Amount"].sum(), 2))
print("Average Account Balance:", round(df["Account_Balance"].mean(), 2))
print("Customers With Loans:", len(loan_customers))
print("Total Loan Amount:", round(loan_customers["Loan_Amount"].sum(), 2))
transactions_by_type.to_csv("analysis_transactions_by_type.csv")
transactions_by_branch.to_csv("analysis_transactions_by_branch.csv")
customer_activity.to_csv("analysis_customer_activity.csv")
account_analysis.to_csv("analysis_account_types.csv")
loan_analysis.to_csv("analysis_loans.csv")
loan_status_analysis.to_csv("analysis_loan_status.csv")
category_analysis.to_csv("analysis_transaction_categories.csv")
balance_status_analysis.to_csv("analysis_balance_status.csv")
print("\nETL COMPLETED SUCCESSFULLY")
print("Extract: banking_dataset.csv")
print("Transform: banking_dataset_cleaned.csv and banking_dataset_transformed.csv")
print("Load: analysis CSV files created")