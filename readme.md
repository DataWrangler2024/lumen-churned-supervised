
## How to Reproduce

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

See `data/README.md` for dataset access instructions. Expected files:

- `X_train.parquet`, `X_test.parquet` — Feature matrices
- `y_train.parquet`, `y_test.parquet` — Target labels (churn or watch hours)

### 3. Run the Pipeline

**Option A: Full pipeline (all models)**

```bash
cd C:\Users\keith.frost\Documents\Python\lumen-churned-supervised
python src\pipeline.py
```

**Option B: Single model**

```bash
python src\pipeline.py --model rf
```

**Option C: Custom data/output paths**

```bash
python src\pipeline.py --data_dir ./data --output_dir ./results
```

### 4. Run Notebooks (Optional)

Open and execute notebooks in order:

1. `notebooks/01_churn_classification.ipynb`
2. `notebooks/02_watch_hour_regression.ipynb`

## Key Results

### Churn Classification

| Model | PR-AUC (10-fold CV) | Best K | Precision@K | Net Profit/Month |
|-------|---------------------|--------|-------------|------------------|
| Logistic Regression (C=0.01) | 0.2899 ± 0.0287 | 1500 | 24.67% | **$3,600** |
| Logistic Regression (C=1.0) | 0.2897 ± 0.0284 | 1500 | 24.67% | $3,600 |
| Random Forest | 0.2636 ± 0.0250 | 1500 | 23.33% | $3,000 |

- **Baseline (no model):** $0 profit (target all or none)
- **Selected model:** LR (C=0.01) — simpler, higher profit at scale, production-ready
- **Statistical significance:** LR vs RF paired t-test, t=6.71, p=0.0001

### Watch-Hour Regression

| Metric | Value |
|--------|-------|
| Model | Ridge Regression (log-transformed target) |
| MAE | 3.39 hours |
| R² | 0.782 |
| Top predictor | Last-month watch hours (corr = 0.88) |

## What I'd Do Next

- **Feature engineering:** Add session-level features (time-of-day, content category mix, device type)
- **Model upgrade:** Test gradient-boosting models (XGBoost, LightGBM) in next iteration
- **Validation:** Build a temporal holdout test on the following month to verify no concept drift
- **Calibration:** Add probability calibration (Platt scaling or isotonic) for better threshold selection

## Design Tradeoffs

| Decision | Rationale | Tradeoff |
|----------|-----------|----------|
| **Ridge over plain LR** | Handles collinear lag features (e.g., multiple watch-hour metrics) | Slight bias for reduced variance |
| **log1p transform on regression target** | Watch hours are highly right-skewed; log reduces outlier influence | Predictions require inverse transform; interpretability cost |
| **Top-K threshold (not 0.5)** | Aligns model output with actual retention team capacity (e.g., can contact ~1500 users/month) | Requires business calibration; not a universal threshold |
| **PR-AUC over ROC-AUC** | Churn is imbalanced (~15%); PR-AUC better reflects performance on the positive class | Less familiar to non-technical stakeholders |
| **Logistic Regression as champion** | Higher profit at scale, faster inference, easier compliance/audit | Slightly lower raw discrimination than RF |

## Outputs Generated

Running `pipeline.py` produces:

- `output/figures/cv_boxplot.png` — Cross-validation score distributions
- `output/cost_benefit_results.csv` — Profit & precision by model × K
- Console output with CV scores, t-test results, and cost-benefit table

## Notes

- All random seeds fixed at `42` for reproducibility
- 10-fold stratified CV ensures class balance in each fold
- Cost-benefit assumptions: $5/intervention, $30 value per saved churner (adjust in `net_profit()` as needed)

---
