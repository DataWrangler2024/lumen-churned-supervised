import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Change to the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv("data/lumen_watch_target.csv")

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

# Plot histogram of watch_hours_next_month
plt.hist(df["watch_hours_next_month"], bins=30, edgecolor="black")
plt.xlabel("Watch Hours Next Month")
plt.ylabel("Frequency")
plt.title("Distribution of Watch Hours Next Month")
plt.savefig("figures/hist_watch_hours.png")
plt.close()

# Plot scatter of each watch_hours_t_minus_*
lag_cols = [col for col in df.columns if col.startswith("watch_hours_t_minus_")]
for col in lag_cols:
    plt.scatter(df[col], df["watch_hours_next_month"], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel("Watch Hours Next Month")
    plt.title(f"Scatter Plot: {col} vs Watch Hours Next Month")
    plt.savefig(f"figures/scatter_{col}.png")
    plt.close()

# Correlation heatmap of lag features and target
sns.heatmap(df[lag_cols + ["watch_hours_next_month"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap: Lag Features vs Target")
plt.savefig("figures/correlation_heatmap.png")
plt.close()

# Log transform
df["watch_hours_next_month_log"] = np.log1p(df["watch_hours_next_month"])

# Train/test split - RAW TARGET
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    df[lag_cols], df["watch_hours_next_month"], test_size=0.2, random_state=42
)

# Train/test split - LOG TARGET
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    df[lag_cols], df["watch_hours_next_month_log"], test_size=0.2, random_state=42
)

# Define models
lr_pipe = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
ridge_pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
rf = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)

# ============ TRAIN ON RAW TARGET ============
print("\n" + "="*60)
print("TRAINING ON RAW TARGET")
print("="*60)

lr_pipe_raw = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
ridge_pipe_raw = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
rf_raw = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)

lr_pipe_raw.fit(X_train_raw, y_train_raw)
ridge_pipe_raw.fit(X_train_raw, y_train_raw)
rf_raw.fit(X_train_raw, y_train_raw)

y_pred_lr_raw = lr_pipe_raw.predict(X_test_raw)
y_pred_ridge_raw = ridge_pipe_raw.predict(X_test_raw)
y_pred_rf_raw = rf_raw.predict(X_test_raw)

# ============ TRAIN ON LOG TARGET ============
print("\n" + "="*60)
print("TRAINING ON LOG-TRANSFORMED TARGET")
print("="*60)

lr_pipe_log = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
ridge_pipe_log = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
rf_log = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)

lr_pipe_log.fit(X_train_log, y_train_log)
ridge_pipe_log.fit(X_train_log, y_train_log)
rf_log.fit(X_train_log, y_train_log)

# Predict on log scale, then transform back to original units
y_pred_lr_log = lr_pipe_log.predict(X_test_log)
y_pred_ridge_log = ridge_pipe_log.predict(X_test_log)
y_pred_rf_log = rf_log.predict(X_test_log)

# Back-transform to original scale
y_pred_lr_log_back = np.expm1(y_pred_lr_log)
y_pred_ridge_log_back = np.expm1(y_pred_ridge_log)
y_pred_rf_log_back = np.expm1(y_pred_rf_log)

# ============ METRICS FUNCTION ============
def report_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = y_true > 1
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    max_residual = np.max(np.abs(y_true - y_pred))
    return {
        "Model": name, 
        "MAE": mae, 
        "RMSE": rmse, 
        "R²": r2, 
        "MAPE": mape,
        "Max |Residual|": max_residual
    }

# ============ COMPARE ALL MODELS ============
print("\n" + "="*60)
print("MODEL COMPARISON: RAW vs LOG TARGET")
print("="*60)

# Raw target models
metrics_lr_raw = report_metrics("Linear Regression (Raw)", y_test_raw, y_pred_lr_raw)
metrics_ridge_raw = report_metrics("Ridge (Raw)", y_test_raw, y_pred_ridge_raw)
metrics_rf_raw = report_metrics("Random Forest (Raw)", y_test_raw, y_pred_rf_raw)

# Log target models (back-transformed predictions)
metrics_lr_log = report_metrics("Linear Regression (Log)", y_test_raw, y_pred_lr_log_back)
metrics_ridge_log = report_metrics("Ridge (Log)", y_test_raw, y_pred_ridge_log_back)
metrics_rf_log = report_metrics("Random Forest (Log)", y_test_raw, y_pred_rf_log_back)

all_metrics = [
    metrics_lr_raw, metrics_ridge_raw, metrics_rf_raw,
    metrics_lr_log, metrics_ridge_log, metrics_rf_log
]

metrics_df = pd.DataFrame(all_metrics)
print("\nModel Performance Summary (all metrics in original units - hours):")
print(metrics_df.to_string(index=False))

# Save metrics table
metrics_df.to_csv("figures/model_metrics_comparison.csv", index=False)

# ============ VISUAL COMPARISON ============
# Bar chart: R² comparison
plt.figure(figsize=(10, 6))
plt.barh(metrics_df["Model"], metrics_df["R²"])
plt.xlabel("R² Score")
plt.title("R² Comparison: Raw vs Log-Transformed Target")
plt.axvline(x=metrics_df["R²"].max(), color='r', linestyle='--', alpha=0.5, label=f"Best: {metrics_df['R²'].max():.3f}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/r2_comparison.png")
plt.close()

# Bar chart: MAE comparison
plt.figure(figsize=(10, 6))
plt.barh(metrics_df["Model"], metrics_df["MAE"])
plt.xlabel("MAE (hours)")
plt.title("MAE Comparison: Raw vs Log-Transformed Target")
plt.axvline(x=metrics_df["MAE"].min(), color='g', linestyle='--', alpha=0.5, label=f"Best: {metrics_df['MAE'].min():.2f}h")
plt.legend()
plt.tight_layout()
plt.savefig("figures/mae_comparison.png")
plt.close()

# ============ BEST MODEL SELECTION ============
best_idx = metrics_df["R²"].idxmax()
best_model_row = metrics_df.loc[best_idx]
best_model_name = best_model_row["Model"]
best_model_r2 = best_model_row["R²"]
best_model_mae = best_model_row["MAE"]
best_model_max_res = best_model_row["Max |Residual|"]

print("\n" + "="*60)
print("BEST MODEL SELECTION")
print("="*60)
print(f"\nWinner: {best_model_name}")
print(f"  - R²: {best_model_r2:.3f}")
print(f"  - MAE: {best_model_mae:.2f} hours")
print(f"  - Max |Residual|: {best_model_max_res:.2f} hours")

# Determine if log or raw won
if "Log" in best_model_name:
    print("\n✓ Log transformation improved performance")
    # Use the log-trained model for final predictions
    if "Linear Regression" in best_model_name:
        final_model = lr_pipe_log
        y_pred_final = y_pred_lr_log_back
    elif "Ridge" in best_model_name:
        final_model = ridge_pipe_log
        y_pred_final = y_pred_ridge_log_back
    else:
        final_model = rf_log
        y_pred_final = y_pred_rf_log_back
else:
    print("\n✗ Raw target performed better (no transformation needed)")
    # Use the raw-trained model for final predictions
    if "Linear Regression" in best_model_name:
        final_model = lr_pipe_raw
        y_pred_final = y_pred_lr_raw
    elif "Ridge" in best_model_name:
        final_model = ridge_pipe_raw
        y_pred_final = y_pred_ridge_raw
    else:
        final_model = rf_raw
        y_pred_final = y_pred_rf_raw

# ============ DIAGNOSTIC PLOTS FOR BEST MODEL ============
# Predicted vs Actual
plt.scatter(y_test_raw, y_pred_final, alpha=0.3, s=4)
plt.plot([0, y_test_raw.max()], [0, y_test_raw.max()], "r--")
plt.xlabel("Actual watch hours")
plt.ylabel("Predicted watch hours")
plt.title(f"Predicted vs Actual — Best Model: {best_model_name}")
plt.savefig("figures/pred_vs_actual.png")
plt.close()

# Residual plot for best model
residuals = y_test_raw - y_pred_final
plt.hist(residuals, bins=60, edgecolor="black")
plt.axvline(0, color="k", linestyle="--")
plt.title(f"Residuals — Best Model: mean={residuals.mean():.2f}, std={residuals.std():.2f}")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.savefig("figures/residuals.png")
plt.close()

# Residuals vs Predicted (check heteroscedasticity)
plt.scatter(y_pred_final, residuals, alpha=0.3, s=4)
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("Predicted watch hours")
plt.ylabel("Residual")
plt.title(f"Residuals vs Predicted — {best_model_name}")
plt.savefig("figures/residuals_vs_predicted_best.png")
plt.close()

# Feature importance (if Random Forest won)
if "Random Forest" in best_model_name:
    feat_imp = pd.DataFrame({
        "Feature": lag_cols,
        "Importance": final_model.named_steps["model"].feature_importances_
    }).sort_values("Importance", ascending=False)
    
    print("\nTop 5 most important lag features:")
    print(feat_imp.head(5).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp["Feature"].head(10), feat_imp["Importance"].head(10))
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Feature Importances — Random Forest")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("figures/feature_importance_best.png")
    plt.close()

# ============ RECOMMENDATION MEMO ============
print("\n" + "="*60)
print("RECOMMENDATION:")
print("="*60)

print(f"""
SHIP THIS MODEL: {best_model_name}

REASON: Highest R² ({best_model_r2:.3f}) among all candidates, indicating best 
explanatory power for variance in next-month watch hours.

KEY METRICS (on held-out test set):
  • MAE: {best_model_mae:.2f} hours (typical prediction error)
  • RMSE: {best_model_row['RMSE']:.2f} hours (penalizes large errors)
  • Max |Residual|: {best_model_max_res:.2f} hours (worst-case observed error)
  • MAPE: {best_model_row['MAPE']:.1f}% (percentage error for business context)

CAVEATS - CONDITIONS THAT WOULD DEGRADE PERFORMANCE:
  1. New users with no viewing history → all lag features missing/zero
  2. Mass content launches or viral events → historical patterns break
  3. Holidays and seasonal anomalies → atypical viewing behavior
  4. Platform outages or data pipeline gaps → missing/corrupted features
  5. Changes in recommendation algorithm → shifts in engagement patterns

MITIGATION STRATEGIES:
  • Add user tenure feature to flag new accounts
  • Incorporate calendar features (holiday flags, weekend indicators)
  • Monitor residual distributions in production; retrain on drift
  • Ensemble with simple baseline (e.g., last-month value) for robustness
  • Consider quantile regression for uncertainty estimates

NEXT STEPS:
  1. Deploy {best_model_name} to staging environment
  2. Run A/B test against current baseline (if any)
  3. Set up monitoring dashboard for MAE/RMSE drift
  4. Schedule quarterly retraining with fresh data
""")

