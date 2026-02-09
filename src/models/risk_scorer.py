"""Deal Risk Scoring Model for SkyGeni Sales Intelligence."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import XGBoost and LightGBM (optional)
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
xgb = None
lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"XGBoost not available: {e}. Using sklearn GradientBoosting instead.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"LightGBM not available: {e}")


class DealRiskScorer:
    """
    Deal Risk Scoring System

    Predicts probability of deal loss to help sales teams prioritize
    at-risk deals and take proactive action.
    """

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.calibrated_model = None
        self.feature_columns: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self.metrics: Dict = {}

    def _create_model(self):
        """Create the base model."""
        if self.model_type == "xgboost":
            if XGBOOST_AVAILABLE:
                return xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="auc",
                    use_label_encoder=False
                )
            else:
                logger.warning("XGBoost not available, using GradientBoostingClassifier")
                return GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    min_samples_split=5,
                    subsample=0.8,
                    random_state=42
                )
        elif self.model_type == "lightgbm":
            if LIGHTGBM_AVAILABLE:
                return lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
            else:
                logger.warning("LightGBM not available, using GradientBoostingClassifier")
                return GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_columns: List[str],
        test_size: float = 0.2,
        calibrate: bool = True
    ) -> Dict:
        """
        Train the deal risk scoring model.

        Args:
            X: Feature dataframe
            y: Target series (1 = Won, 0 = Lost)
            feature_columns: List of feature column names
            test_size: Fraction for test split
            calibrate: Whether to calibrate probabilities

        Returns:
            Dictionary of evaluation metrics
        """
        self.feature_columns = feature_columns
        X_features = X[feature_columns]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"Training {self.model_type} model...")
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # Calibrate probabilities for better risk estimates
        if calibrate:
            self.calibrated_model = CalibratedClassifierCV(
                self.model, cv=5, method="isotonic"
            )
            self.calibrated_model.fit(X_train, y_train)

        # Evaluate
        self.metrics = self._evaluate(X_test, y_test)

        # Feature importance
        self._calculate_feature_importance()

        logger.info(f"Model trained. AUC: {self.metrics['auc']:.4f}")
        return self.metrics

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance."""
        y_pred = self.model.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        return metrics

    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance."""
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            importance = np.zeros(len(self.feature_columns))

        self.feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importance
        }).sort_values("importance", ascending=False)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using calibrated model if available."""
        X_features = X[self.feature_columns] if isinstance(X, pd.DataFrame) else X

        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_features)
        return self.model.predict_proba(X_features)

    def score_deals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score deals with risk probability and category.

        Returns dataframe with:
        - risk_score: Probability of LOSS (1 - win probability)
        - risk_category: High/Medium/Low risk label
        - win_probability: Probability of winning
        """
        df = df.copy()

        # Get win probability
        probs = self.predict_proba(df[self.feature_columns])
        df["win_probability"] = probs[:, 1]

        # Risk score is inverse of win probability
        df["risk_score"] = 1 - df["win_probability"]

        # Categorize risk
        df["risk_category"] = pd.cut(
            df["risk_score"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        )

        return df

    def get_risk_factors(self, deal: pd.Series) -> Dict:
        """
        Get top risk factors for a specific deal using feature importance.

        For true explainability, use SHAP (see evaluation/explainability.py)
        """
        factors = []
        for _, row in self.feature_importance.head(5).iterrows():
            feature = row["feature"]
            value = deal.get(feature, "N/A")
            factors.append({
                "feature": feature,
                "value": value,
                "importance": row["importance"]
            })
        return factors

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """Perform cross-validation."""
        X_features = X[self.feature_columns]

        cv_scores = cross_val_score(
            self._create_model(), X_features, y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring="roc_auc"
        )

        return {
            "cv_scores": cv_scores.tolist(),
            "mean_auc": cv_scores.mean(),
            "std_auc": cv_scores.std()
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, f"{path}/model.joblib")
        if self.calibrated_model:
            joblib.dump(self.calibrated_model, f"{path}/calibrated_model.joblib")
        joblib.dump(self.feature_columns, f"{path}/feature_columns.joblib")
        joblib.dump(self.metrics, f"{path}/metrics.joblib")

        if self.feature_importance is not None:
            self.feature_importance.to_csv(f"{path}/feature_importance.csv", index=False)

        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(f"{path}/model.joblib")
        self.feature_columns = joblib.load(f"{path}/feature_columns.joblib")
        self.metrics = joblib.load(f"{path}/metrics.joblib")

        calibrated_path = f"{path}/calibrated_model.joblib"
        if Path(calibrated_path).exists():
            self.calibrated_model = joblib.load(calibrated_path)

        importance_path = f"{path}/feature_importance.csv"
        if Path(importance_path).exists():
            self.feature_importance = pd.read_csv(importance_path)

        logger.info(f"Model loaded from {path}")


class EnsembleRiskScorer:
    """Ensemble of multiple models for robust risk scoring."""

    def __init__(self):
        self.models = {
            "xgboost": DealRiskScorer("xgboost"),
            "lightgbm": DealRiskScorer("lightgbm"),
            "random_forest": DealRiskScorer("random_forest")
        }
        self.weights = {"xgboost": 0.4, "lightgbm": 0.4, "random_forest": 0.2}
        self.feature_columns: List[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series, feature_columns: List[str]) -> Dict:
        """Train all models in the ensemble."""
        self.feature_columns = feature_columns
        all_metrics = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            metrics = model.train(X, y, feature_columns)
            all_metrics[name] = metrics

        return all_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of all model predictions."""
        X_features = X[self.feature_columns]
        weighted_probs = np.zeros((len(X), 2))

        for name, model in self.models.items():
            probs = model.predict_proba(X_features)
            weighted_probs += probs * self.weights[name]

        return weighted_probs

    def score_deals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score deals using ensemble."""
        df = df.copy()
        probs = self.predict_proba(df)

        df["win_probability"] = probs[:, 1]
        df["risk_score"] = 1 - df["win_probability"]
        df["risk_category"] = pd.cut(
            df["risk_score"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        )

        return df
