from gettext import install
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import sys

#Change to the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#import csv
df = pd.read_csv('data/lumen_subscriptions.csv')

#Summary stats
print(round(df.describe(), 2))
print("\n" + str(df.isna().sum()))

print(df["churned_30d"].mean())     # base rate
print(df["churned_30d"].value_counts(normalize=True))

#plot churned_30d distribution
ax = df["churned_30d"].value_counts(normalize=True).plot(
    kind="bar",
    title="Churned 30 Days Distribution"
)
plt.ylabel("Proportion")
plt.xlabel("churned_30d")
plt.tight_layout()
plt.show()

#save plot
save_path = r'C:\Users\keith.frost\Documents\Python\lumen-churned-supervised\figures\churn_30d_distribution.png'
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Saved plot to: {save_path}")

#Group by churned_30d and calculate mean for each numeric feature
print("Numeric feature means by churned_30d:")

print("\nChurn rate by user_id:")
print(
    df.groupby("user_id")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

print("\nChurn rate by monthly_price_eur:")
print(
    df.groupby("monthly_price_eur")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

print("\nChurn rate by tenure_months:")
print(
    df.groupby("tenure_months")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

print("\nChurn rate by monthly_watch_hours_90d:")
print(
    df.groupby("monthly_watch_hours_90d")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

print("\nChurn rate by sessions_90d:")
print(
    df.groupby("sessions_90d")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

print("\nChurn rate by unique_titles_90d:")
print(
    df.groupby("unique_titles_90d")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

print("\nChurn rate by support_tickets_90d:")
print(
    df.groupby("support_tickets_90d")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

#group by churned_30d and calculate mean for each categorical feature
print("\nChurn rate by payment_method:")
print(
    df.groupby("payment_method")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

print("\nChurn rate by plan:")
print(
    df.groupby("plan")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

print("\nChurn rate by signup_channel:")
print(
    df.groupby("signup_channel")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

print("\nChurn rate by country:")
print(
    df.groupby("country")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

#Create new features
df["watch_per_session"] = df["monthly_watch_hours_90d"] / df["sessions_90d"].clip(lower=1)
df["is_low_engagement"] = (df["monthly_watch_hours_90d"] < 2).astype(int)
df["price_per_watch_hour"] = (df["monthly_price_eur"] * 3) / df["monthly_watch_hours_90d"].clip(lower=0.5)


print("\nChurn rate by avg_watch_hours_per_session:")
#justification for feature engineering: if a user watches a lot of hours per session, they are likely more engaged and less likely to churn. Conversely, if a user watches very few hours per session, they may be less engaged and more likely to churn.
print(
    df.groupby("watch_per_session")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)


print("\nChurn rate by is_low_engagement:")
#justification for feature engineering: if a user has low engagement (less than 2 hours of watch time per month), they are likely more likely to churn. Conversely, if a user has high engagement (more than 2 hours of watch time per month), they are likely less likely to churn.
print(
    df.groupby("is_low_engagement")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

print("\nChurn rate by price_per_watch_hour:")
#justification for feature engineering: if a user is paying a high price per watch hour, they may be more likely to churn. Conversely, if a user is paying a low price per watch hour, they may be less likely to churn.
print(
    df.groupby("price_per_watch_hour")["churned_30d"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

#Convert categorical features to dummy variables
print("\nDummy variables for categorical features:")
df = pd.get_dummies(df, columns=["plan", "signup_channel", "payment_method"], drop_first=True)

#group countries into top 8 and "OTHER"
top_countries = df["country"].value_counts().head(8).index
df["country_grouped"] = df["country"].where(df["country"].isin(top_countries), other="OTHER")

#One-hot encode the grouped country variable and drop the original country column
df = pd.get_dummies(df, columns=["country_grouped"], drop_first=True)
df = df.drop(columns=["country"])

#Split the data into training and test sets
y = df["churned_30d"]
X = df.drop(columns=["churned_30d", "user_id"])   # drop label and ID

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#print the size of the train and test sets and the churn rate in each
print(f"Train: {len(X_train)}, churn rate: {y_train.mean():.3f}")
print(f"Test:  {len(X_test)}, churn rate: {y_test.mean():.3f}")

#save the splits so later steps load them directly (and can never peek at the test set):
#write the train and test sets to parquet files
os.makedirs("C:\\Users\\keith.frost\\Documents\\Python\\lumen-churned-supervised\\data", exist_ok=True)
X_train.to_parquet("C:\\Users\\keith.frost\\Documents\\Python\\lumen-churned-supervised\\data\\X_train.parquet")
X_test.to_parquet("C:\\Users\\keith.frost\\Documents\\Python\\lumen-churned-supervised\\data\\X_test.parquet")
y_train.to_frame().to_parquet("C:\\Users\\keith.frost\\Documents\\Python\\lumen-churned-supervised\\data\\y_train.parquet")
y_test.to_frame().to_parquet("C:\\Users\\keith.frost\\Documents\\Python\\lumen-churned-supervised\\data\\y_test.parquet")