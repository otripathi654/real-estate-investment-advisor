"""
=============================================================
 Real Estate Investment Advisor
 Step 3: Model Training + MLflow Experiment Tracking
=============================================================
 Usage:
   python train_models.py --input data/processed_data.csv
                          --model_dir models/
                          --experiment_name RealEstate_Advisor

 What this script does:
   1. Prepares feature matrix & both target variables
   2. Trains & evaluates CLASSIFICATION models
      - Logistic Regression (baseline)
      - Random Forest Classifier
      - XGBoost Classifier  ← best logged to MLflow
   3. Trains & evaluates REGRESSION models
      - Linear Regression (baseline)
      - Random Forest Regressor
      - XGBoost Regressor  ← best logged to MLflow
   4. Logs all runs (params, metrics, plots, models) to MLflow
   5. Saves best models as .pkl for Streamlit app
   6. Prints a final comparison summary
=============================================================
"""

import argparse
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model      import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble          import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics           import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.preprocessing     import StandardScaler
from xgboost                   import XGBClassifier, XGBRegressor
import mlflow
import mlflow.sklearn
import mlflow.xgboost

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
CV_FOLDS        = 5
TARGET_CLF      = "Good_Investment"
TARGET_REG      = "Future_Price_5yr"

# Columns to always drop before training
DROP_COLS = [
    "ID", "State", "City", "Locality",
    "Property_Type", "Furnished_Status", "Facing",
    "Owner_Type", "Availability_Status", "Security",
    "Amenities", "Public_Transport_Accessibility",
    "Year_Built",
    TARGET_CLF, TARGET_REG,   # targets themselves
]


# ─────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """Return X (feature matrix) ready for modelling."""
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")

    # Keep only numeric columns (encoded features are already numeric)
    X = X.select_dtypes(include=[np.number])

    # Drop columns that are all-NaN
    X = X.dropna(axis=1, how="all")

    # Fill any remaining NaNs with column median
    X = X.fillna(X.median(numeric_only=True))

    print(f"  Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Features       : {list(X.columns)[:10]} … (first 10 shown)")
    return X


def get_targets(df: pd.DataFrame):
    clf_target = df[TARGET_CLF] if TARGET_CLF in df.columns else None
    reg_target = df[TARGET_REG] if TARGET_REG in df.columns else None
    if clf_target is None:
        raise ValueError(f"Target column '{TARGET_CLF}' not found. Run preprocessing first.")
    if reg_target is None:
        raise ValueError(f"Target column '{TARGET_REG}' not found. Run preprocessing first.")
    return clf_target, reg_target


# ─────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Good", "Good"],
                yticklabels=["Not Good", "Good"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return save_path


def plot_feature_importance(model, feature_names, title, save_path, top_n=20):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        return None

    idx   = np.argsort(imp)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = imp[idx]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    ax.barh(names, vals, color="#2563EB", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return save_path


def plot_actual_vs_predicted(y_true, y_pred, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.3, s=15, color="#2563EB")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()

    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=50, color="#7C3AED", alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Residual (Actual − Predicted)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────
# CLASSIFICATION TRAINING
# ─────────────────────────────────────────────

def train_classification(X_train, X_test, y_train, y_test,
                         feature_names, model_dir, experiment_name):
    print("\n" + "="*55)
    print("  CLASSIFICATION  →  Good_Investment")
    print("="*55)

    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0
        ),
        "Random_Forest_Clf": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost_Clf": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        ),
    }

    results   = {}
    best_name = None
    best_f1   = -1

    mlflow.set_experiment(f"{experiment_name}_Classification")

    for name, model in models.items():
        print(f"\n  ▶ Training {name} …")
        with mlflow.start_run(run_name=name):

            # ── Train ──────────────────────────────────
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # ── Metrics ────────────────────────────────
            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                   random_state=RANDOM_STATE),
                scoring="f1_weighted", n_jobs=-1
            )

            print(f"     Accuracy  : {acc:.4f}")
            print(f"     F1-Score  : {f1:.4f}")
            print(f"     Precision : {prec:.4f}")
            print(f"     Recall    : {rec:.4f}")
            print(f"     CV F1     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"\n{classification_report(y_test, y_pred, target_names=['Not Good','Good Inv.'])}")

            # ── Log to MLflow ──────────────────────────
            mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
            mlflow.log_metrics({
                "accuracy":  acc,
                "f1_weighted": f1,
                "precision": prec,
                "recall":    rec,
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std":  cv_scores.std(),
            })

            # Confusion matrix artifact
            cm_path = os.path.join(model_dir, f"cm_{name}.png")
            plot_confusion_matrix(y_test, y_pred,
                                  f"Confusion Matrix — {name}", cm_path)
            mlflow.log_artifact(cm_path)

            # Feature importance artifact
            fi_path = os.path.join(model_dir, f"fi_{name}.png")
            p = plot_feature_importance(model, feature_names,
                                        f"Feature Importance — {name}", fi_path)
            if p:
                mlflow.log_artifact(fi_path)

            # Log model
            if "XGBoost" in name:
                mlflow.xgboost.log_model(model, artifact_path="model",
                                         registered_model_name=f"RealEstate_{name}")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model",
                                         registered_model_name=f"RealEstate_{name}")

            results[name] = {"accuracy": acc, "f1": f1,
                             "precision": prec, "recall": rec,
                             "cv_f1": cv_scores.mean(), "model": model}

            if f1 > best_f1:
                best_f1   = f1
                best_name = name

    # ── Save best model ────────────────────────────────────
    best_model = results[best_name]["model"]
    best_path  = os.path.join(model_dir, "best_classifier.pkl")
    joblib.dump(best_model, best_path)
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.pkl"))
    print(f"\n  🏆  Best Classifier : {best_name}  (F1 = {best_f1:.4f})")
    print(f"  💾  Saved to        : {best_path}")

    return results, best_name


# ─────────────────────────────────────────────
# REGRESSION TRAINING
# ─────────────────────────────────────────────

def train_regression(X_train, X_test, y_train, y_test,
                     feature_names, model_dir, experiment_name):
    print("\n" + "="*55)
    print("  REGRESSION  →  Future_Price_5yr")
    print("="*55)

    models = {
        "Ridge_Regression": Ridge(alpha=10.0),
        "Random_Forest_Reg": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost_Reg": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        ),
    }

    results   = {}
    best_name = None
    best_r2   = -np.inf

    mlflow.set_experiment(f"{experiment_name}_Regression")

    for name, model in models.items():
        print(f"\n  ▶ Training {name} …")
        with mlflow.start_run(run_name=name):

            # ── Train ──────────────────────────────────
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # ── Metrics ────────────────────────────────
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100

            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=CV_FOLDS, scoring="r2", n_jobs=-1
            )

            print(f"     RMSE  : {rmse:.4f}")
            print(f"     MAE   : {mae:.4f}")
            print(f"     R²    : {r2:.4f}")
            print(f"     MAPE  : {mape:.2f}%")
            print(f"     CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # ── Log to MLflow ──────────────────────────
            mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
            mlflow.log_metrics({
                "RMSE":    rmse,
                "MAE":     mae,
                "R2":      r2,
                "MAPE":    mape,
                "cv_r2_mean": cv_scores.mean(),
                "cv_r2_std":  cv_scores.std(),
            })

            # Actual vs Predicted artifact
            avp_path = os.path.join(model_dir, f"avp_{name}.png")
            plot_actual_vs_predicted(y_test, y_pred,
                                     f"Actual vs Predicted — {name}", avp_path)
            mlflow.log_artifact(avp_path)

            # Feature importance artifact
            fi_path = os.path.join(model_dir, f"fi_reg_{name}.png")
            p = plot_feature_importance(model, feature_names,
                                        f"Feature Importance — {name}", fi_path)
            if p:
                mlflow.log_artifact(fi_path)

            # Log model
            if "XGBoost" in name:
                mlflow.xgboost.log_model(model, artifact_path="model",
                                         registered_model_name=f"RealEstate_{name}")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model",
                                         registered_model_name=f"RealEstate_{name}")

            results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2,
                             "MAPE": mape, "cv_r2": cv_scores.mean(),
                             "model": model}

            if r2 > best_r2:
                best_r2   = r2
                best_name = name

    # ── Save best model ────────────────────────────────────
    best_model = results[best_name]["model"]
    best_path  = os.path.join(model_dir, "best_regressor.pkl")
    joblib.dump(best_model, best_path)
    print(f"\n  🏆  Best Regressor : {best_name}  (R² = {best_r2:.4f})")
    print(f"  💾  Saved to       : {best_path}")

    return results, best_name


# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────

def print_summary(clf_results, best_clf, reg_results, best_reg):
    print("\n" + "="*55)
    print("  FINAL MODEL COMPARISON SUMMARY")
    print("="*55)

    print("\n  📊  Classification (Good_Investment)")
    print(f"  {'Model':<30} {'Accuracy':>9} {'F1':>8} {'CV-F1':>8}")
    print("  " + "-"*58)
    for name, m in clf_results.items():
        star = " ⭐" if name == best_clf else ""
        print(f"  {name:<30} {m['accuracy']:>9.4f} {m['f1']:>8.4f} {m['cv_f1']:>8.4f}{star}")

    print("\n  📈  Regression (Future_Price_5yr)")
    print(f"  {'Model':<30} {'RMSE':>10} {'MAE':>10} {'R²':>8} {'CV-R²':>8}")
    print("  " + "-"*68)
    for name, m in reg_results.items():
        star = " ⭐" if name == best_reg else ""
        print(f"  {name:<30} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['R2']:>8.4f} {m['cv_r2']:>8.4f}{star}")

    print("\n  📁  Saved model files:")
    print("      models/best_classifier.pkl  ← use in Streamlit app")
    print("      models/best_regressor.pkl   ← use in Streamlit app")
    print("      models/feature_names.pkl    ← column list for inference")
    print("\n  🔬  MLflow UI:")
    print("      mlflow ui  (then open http://localhost:5000)")
    print("="*55 + "\n")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_training(input_path: str, model_dir: str, experiment_name: str):
    os.makedirs(model_dir, exist_ok=True)

    # ── Load data ─────────────────────────────
    print(f"\n{'='*55}")
    print("  Loading processed dataset …")
    df = pd.read_csv(input_path)
    print(f"  Shape: {df.shape}")

    # ── Prepare features & targets ────────────
    print("\n[Feature Preparation]")
    X             = prepare_features(df)
    y_clf, y_reg  = get_targets(df)
    feature_names = list(X.columns)

    # ── Train / test split ────────────────────
    X_train, X_test, yc_train, yc_test = train_test_split(
        X, y_clf, test_size=TEST_SIZE, stratify=y_clf, random_state=RANDOM_STATE
    )
    _, _, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n  Train size : {len(X_train):,}  |  Test size : {len(X_test):,}")

    # ── Set MLflow tracking URI ───────────────
    mlflow.set_tracking_uri("mlflow_runs/")

    # ── Train classification models ───────────
    clf_results, best_clf = train_classification(
        X_train, X_test, yc_train, yc_test,
        feature_names, model_dir, experiment_name
    )

    # ── Train regression models ───────────────
    reg_results, best_reg = train_regression(
        X_train, X_test, yr_train, yr_test,
        feature_names, model_dir, experiment_name
    )

    # ── Save metadata for Streamlit ──────────
    meta = {
        "feature_names":    feature_names,
        "target_clf":       TARGET_CLF,
        "target_reg":       TARGET_REG,
        "best_classifier":  best_clf,
        "best_regressor":   best_reg,
        "clf_metrics":      {k: {m: v for m, v in v.items() if m != "model"}
                             for k, v in clf_results.items()},
        "reg_metrics":      {k: {m: v for m, v in v.items() if m != "model"}
                             for k, v in reg_results.items()},
    }
    joblib.dump(meta, os.path.join(model_dir, "model_metadata.pkl"))

    # ── Print summary ─────────────────────────
    print_summary(clf_results, best_clf, reg_results, best_reg)

    return clf_results, reg_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Estate — Model Training")
    parser.add_argument("--input",           default="data/processed_data.csv")
    parser.add_argument("--model_dir",       default="models/")
    parser.add_argument("--experiment_name", default="RealEstate_Advisor")
    args = parser.parse_args()
    run_training(args.input, args.model_dir, args.experiment_name)
