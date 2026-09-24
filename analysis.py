import pandas as pd
import numpy as np
df = pd.read_csv("dataset_transformed.csv")
df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    errors="coerce"
)
print("\nDATASET SHAPE")
print(df.shape)
print("\nDATASET INFORMATION")
df.info()
print("\n1. OVERALL BANKING METRICS")
total_transactions = len(df)
total_amount = df["Transaction_Amount"].sum()
average_amount = df["Transaction_Amount"].mean()
maximum_amount = df["Transaction_Amount"].max()
minimum_amount = df["Transaction_Amount"].min()
median_amount = np.median(df["Transaction_Amount"].dropna())
total_customers = df["Customer_ID"].nunique()
total_accounts = df["Account_Number"].nunique()
total_branches = df["Branch"].nunique()
print("Total Transactions:", total_transactions)
print("Total Transaction Amount:", total_amount)
print("Average Transaction Amount:", average_amount)
print("Maximum Transaction Amount:", maximum_amount)
print("Minimum Transaction Amount:", minimum_amount)
print("Median Transaction Amount:", median_amount)
print("Total Unique Customers:", total_customers)
print("Total Unique Accounts:", total_accounts)
print("Total Branches:", total_branches)
print("\n2. TRANSACTIONS BY TYPE")
transactions_by_type = df.groupby(
    "Transaction_Type"
).agg(
    Transaction_Count=("Transaction_Amount", "count"),
    Total_Amount=("Transaction_Amount", "sum"),
    Average_Amount=("Transaction_Amount", "mean")
).round(2).reset_index()
print(transactions_by_type)
print("\n3. TRANSACTIONS BY BRANCH")
transactions_by_branch = df.groupby(
    "Branch"
).agg(
    Transaction_Count=("Transaction_Amount", "count"),
    Total_Amount=("Transaction_Amount", "sum"),
    Average_Amount=("Transaction_Amount", "mean"),
    Unique_Customers=("Customer_ID", "nunique")
).round(2).sort_values(
    "Total_Amount", ascending=False
).reset_index()
print(transactions_by_branch)
print("\n4. CUSTOMER ACTIVITY")
customer_activity = df.groupby(
    ["Customer_ID", "Customer_Name"]
).agg(
    Transaction_Count=("Transaction_Amount", "count"),
    Total_Transaction_Amount=("Transaction_Amount", "sum"),
    Average_Transaction_Amount=("Transaction_Amount", "mean"),
    Maximum_Transaction_Amount=("Transaction_Amount", "max")
).round(2).sort_values(
    "Total_Transaction_Amount", ascending=False
).reset_index()
print(customer_activity)
print("\n5. TOP 5 ACTIVE CUSTOMERS")
top_customers = customer_activity.sort_values(
    ["Transaction_Count", "Total_Transaction_Amount"],
    ascending=False
).head(5)
print(top_customers)
print("\n6. ACCOUNT BALANCE ANALYSIS")
account_balances = df.sort_values(
    "Transaction_Date"
).drop_duplicates(
    subset=["Account_Number"],
    keep="last"
)
balance_analysis = account_balances[
    "Account_Balance"
].agg(
    ["sum", "mean", "median", "min", "max"]
).round(2)
print(balance_analysis)
print("\n7. ACCOUNT BALANCE BY ACCOUNT TYPE")
balance_by_account_type = account_balances.groupby(
    "Account_Type"
).agg(
    Total_Accounts=("Account_Number", "nunique"),
    Total_Balance=("Account_Balance", "sum"),
    Average_Balance=("Account_Balance", "mean")
).round(2).reset_index()
print(balance_by_account_type)
print("\n8. BRANCH-WISE ACCOUNT BALANCES")
branch_balances = account_balances.groupby(
    "Branch"
).agg(
    Total_Accounts=("Account_Number", "nunique"),
    Total_Balance=("Account_Balance", "sum"),
    Average_Balance=("Account_Balance", "mean")
).round(2).sort_values(
    "Total_Balance", ascending=False
).reset_index()
print(branch_balances)
print("\n9. TRANSACTION AMOUNT CATEGORY ANALYSIS")
transaction_categories = df.groupby(
    "Transaction_Amount_Category"
).agg(
    Transaction_Count=("Transaction_Amount", "count"),
    Total_Amount=("Transaction_Amount", "sum"),
    Average_Amount=("Transaction_Amount", "mean")
).round(2).reset_index()
print(transaction_categories)
print("\n10. HIGH VALUE TRANSACTION ANALYSIS")
high_value_transactions = df[
    df["High_Value_Transaction_Indicator"] == "Yes"
]
print("High Value Transaction Count:",
      len(high_value_transactions))
print("High Value Transaction Total:",
      high_value_transactions["Transaction_Amount"].sum())
print("\n11. MONTHLY TRANSACTION ANALYSIS")
monthly_transactions = df.dropna(
    subset=["Transaction_Date"]
).copy()
monthly_transactions["Month"] = (
    monthly_transactions["Transaction_Date"]
    .dt.to_period("M")
    .astype(str)
)
monthly_analysis = monthly_transactions.groupby(
    "Month"
).agg(
    Transaction_Count=("Transaction_Amount", "count"),
    Total_Amount=("Transaction_Amount", "sum"),
    Average_Amount=("Transaction_Amount", "mean")
).round(2).reset_index()
print(monthly_analysis)
print("\n12. LOAN ANALYSIS")
loan_columns = [
    col for col in df.columns
    if "loan" in col.lower()
]
if loan_columns:
    print("Available Loan Columns:", loan_columns)

    if "Loan_Amount" in df.columns:
        loan_amount = pd.to_numeric(
            df["Loan_Amount"],
            errors="coerce"
        )
        print("Total Loan Amount:", loan_amount.sum())
        print("Average Loan Amount:", loan_amount.mean())
        print("Maximum Loan Amount:", loan_amount.max())

    if "Loan_Status" in df.columns:
        print("\nLoan Status Analysis")
        print(df["Loan_Status"].value_counts())

else:
    print("Loan information is not available in this dataset.")
print("\n13. DATA VALIDATION")
print("Total Transaction Amount Matches:",
      np.isclose(
          transactions_by_type["Total_Amount"].sum(),
          total_amount
      ))
print("Branch Transaction Count Matches:",
      transactions_by_branch["Transaction_Count"].sum()
      == total_transactions)
print("Customer Transaction Count Matches:",
      customer_activity["Transaction_Count"].sum()
      == total_transactions)
print("\n14. SAVE ANALYSIS RESULTS")
transactions_by_type.to_csv(
    "transactions_by_type.csv", index=False
)
transactions_by_branch.to_csv(
    "transactions_by_branch.csv", index=False
)
customer_activity.to_csv(
    "customer_activity.csv", index=False
)
top_customers.to_csv(
    "top_5_customers.csv", index=False
)
balance_by_account_type.to_csv(
    "balance_by_account_type.csv", index=False
)
branch_balances.to_csv(
    "branch_balances.csv", index=False
)
transaction_categories.to_csv(
    "transaction_categories.csv", index=False
)
high_value_transactions.to_csv(
    "high_value_transactions.csv", index=False
)
monthly_analysis.to_csv(
    "monthly_analysis.csv", index=False
)
print("\nAnalysis Successful")