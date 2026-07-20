#pipeline.py - Reusable ML Training & Evaluation Pipeline

import os
import sys
import argparse
import warnings
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import ttest_rel

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Definitions
# =============================================================================

def get_model_registry() -> Dict[str, Pipeline]:
    """
    Returns a dictionary of model name -> pipeline.
    Add new models here for automatic inclusion in experiments.
    """
    return {
        "lr_c1": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        ),
        "lr_c001": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.01, max_iter=1000, random_state=42)
        ),
        "rf": make_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        ),
    }


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load train/test splits from parquet files.
    
    Parameters
    ----------
    data_dir : str
        Directory containing X_train.parquet, X_test.parquet, y_train.parquet, y_test.parquet
    
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    data_path = Path(data_dir)
    
    X_train = pd.read_parquet(data_path / "X_train.parquet")
    X_test = pd.read_parquet(data_path / "X_test.parquet")
    y_train = pd.read_parquet(data_path / "y_train.parquet").iloc[:, 0]
    y_test = pd.read_parquet(data_path / "y_test.parquet").iloc[:, 0]
    
    logger.info(f"Loaded data: X_train={X_train.shape}, X_test={X_test.shape}")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# Cross-Validation Evaluation
# =============================================================================

def evaluate_models_cv(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 10,
    scoring: str = "average_precision",
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Perform stratified k-fold cross-validation for multiple models.
    
    Parameters
    ----------
    models : dict
        Model name -> pipeline dictionary
    X : DataFrame
        Features
    y : Series
        Target
    n_splits : int
        Number of CV folds
    scoring : str
        Scoring metric (default: "average_precision")
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    results : dict
        Model name -> array of per-fold scores
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}
    
    logger.info(f"Running {n_splits}-fold CV with scoring='{scoring}'...")
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = scores
        logger.info(f"{name}: {scoring} = {scores.mean():.4f} ± {scores.std():.4f}")
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_cv_boxplot(
    results: Dict[str, np.ndarray],
    output_path: str,
    title: str = "Cross-Validation Results",
    ylabel: str = "PR-AUC (per fold)"
) -> None:
    """
    Generate and save boxplot of CV scores.
    
    Parameters
    ----------
    results : dict
        Model name -> per-fold scores
    output_path : str
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot(results.values(), tick_labels=results.keys())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved CV boxplot to {output_path}")


# =============================================================================
# Statistical Testing
# =============================================================================

def paired_ttest(
    results: Dict[str, np.ndarray],
    model_a: str,
    model_b: str
) -> Tuple[float, float]:
    """
    Perform paired t-test between two models' CV scores.
    
    Parameters
    ----------
    results : dict
        Model name -> per-fold scores
    model_a, model_b : str
        Names of models to compare
    
    Returns
    -------
    t_stat, p_value
    """
    scores_a = results[model_a]
    scores_b = results[model_b]
    
    t_stat, p_val = ttest_rel(scores_a, scores_b)
    logger.info(f"Paired t-test ({model_a} vs {model_b}): t={t_stat:.2f}, p={p_val:.4f}")
    
    return t_stat, p_val


# =============================================================================
# Cost-Benefit Analysis
# =============================================================================

def net_profit(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    K: int,
    cost: float = 5.0,
    save_value: float = 30.0
) -> Tuple[float, float]:
    """
    Calculate net profit for targeting top-K customers by predicted churn probability.
    
    Parameters
    ----------
    model : Pipeline
        Trained sklearn pipeline
    X_test : DataFrame
        Test features
    y_test : Series
        True labels
    K : int
        Number of customers to target
    cost : float
        Cost per intervention (default: $5)
    save_value : float
        Value of saving one churner (default: $30)
    
    Returns
    -------
    profit, precision
    """
    probs = model.predict_proba(X_test)[:, 1]
    top_k_idx = np.argsort(probs)[-K:]
    n_real_churners = y_test.iloc[top_k_idx].sum()
    profit = n_real_churners * save_value - K * cost
    precision = n_real_churners / K
    
    return profit, precision


def cost_benefit_analysis(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k_values: List[int] = [200, 400, 800, 1500]
) -> pd.DataFrame:
    """
    Generate cost-benefit comparison table across models and K values.
    
    Parameters
    ----------
    models : dict
        Trained model name -> pipeline dictionary
    X_test, y_test : DataFrame, Series
        Test data
    k_values : list
        List of K values to evaluate
    
    Returns
    -------
    DataFrame with profit and precision for each model × K combination
    """
    results_rows = []
    
    for name, model in models.items():
        row = {"Model": name}
        for K in k_values:
            profit, precision = net_profit(model, X_test, y_test, K)
            row[f"K={K}"] = f"${profit:,.0f} (prec={precision:.2%})"
        results_rows.append(row)
    
    df_results = pd.DataFrame(results_rows)
    return df_results


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(
    data_dir: str = "data",
    output_dir: str = "output",
    model_focus: Optional[str] = None
) -> Dict:
    """
    Execute full ML pipeline: load → evaluate → test → analyze → save.
    
    Parameters
    ----------
    data_dir : str
        Directory with parquet data files
    output_dir : str
        Directory to save outputs
    model_focus : str, optional
        If specified, only run on this model (e.g., "rf")
    
    Returns
    -------
    Dictionary with all results and outputs
    """
    # Setup directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_path = output_path / "figures"
    figures_path.mkdir(exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_dir)
    
    # Get models
    model_registry = get_model_registry()
    if model_focus:
        model_registry = {model_focus: model_registry[model_focus]}
    
    # Cross-validation
    cv_results = evaluate_models_cv(model_registry, X_train, y_train, n_splits=10)
    
    # Plot CV results
    plot_cv_boxplot(
        cv_results,
        output_path=figures_path / "cv_boxplot.png",
        title="10-Fold Cross-Validation: PR-AUC"
    )
    
    # Paired t-test (LR C=1.0 vs RF if both exist)
    t_stat, p_val = None, None
    if "lr_c1" in cv_results and "rf" in cv_results:
        t_stat, p_val = paired_ttest(cv_results, "lr_c1", "rf")
    
    # Train models on full training data for cost-benefit analysis
    trained_models = {}
    for name, model in model_registry.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    # Cost-benefit analysis
    k_values = [200, 400, 800, 1500]
    df_benefit = cost_benefit_analysis(trained_models, X_test, y_test, k_values)
    
    # Save results
    df_benefit.to_csv(output_path / "cost_benefit_results.csv", index=False)
    logger.info(f"Saved cost-benefit results to {output_path / 'cost_benefit_results.csv'}")
    
    # Print summary
    print("\n" + "="*60)
    print("COST-BENEFIT ANALYSIS")
    print("="*60)
    print(df_benefit.to_string(index=False))
    print("="*60)
    
    return {
        "cv_results": cv_results,
        "t_test": (t_stat, p_val),
        "cost_benefit": df_benefit,
        "trained_models": trained_models
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reusable ML Training & Evaluation Pipeline"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing parquet data files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lr_c1", "lr_c001", "rf"],
        default=None,
        help="Run on specific model only (default: all)"
    )
    
    args = parser.parse_args()
    
    results = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_focus=args.model
    )
    
    logger.info("Pipeline completed successfully.")
