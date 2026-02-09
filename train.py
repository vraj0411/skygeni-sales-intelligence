#!/usr/bin/env python3
"""
SkyGeni Sales Intelligence - Model Training Pipeline

This script trains the Deal Risk Scoring model and saves artifacts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import yaml

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor, create_time_features
from src.features.engineering import FeatureEngineer, calculate_pipeline_metrics
from src.models.risk_scorer import DealRiskScorer, EnsembleRiskScorer
from src.evaluation.metrics import ModelEvaluator, calculate_business_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("SkyGeni Sales Intelligence - Training Pipeline")
    logger.info("=" * 60)

    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Data
    logger.info("\n[1/6] Loading data...")
    loader = DataLoader(config["data"]["raw_path"])
    df = loader.load()
    summary = loader.get_summary()

    logger.info(f"Total deals: {summary['total_deals']}")
    logger.info(f"Win rate: {summary['win_rate']:.2%}")
    logger.info(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")

    # 2. Preprocess
    logger.info("\n[2/6] Preprocessing...")
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)
    df = create_time_features(df)

    # 3. Feature Engineering
    logger.info("\n[3/6] Feature engineering...")
    feature_engineer = FeatureEngineer()
    df = feature_engineer.fit_transform(df)

    # Define feature columns for modeling
    base_features = preprocessor.get_feature_columns()
    engineered_features = feature_engineer.get_engineered_features()
    time_features = ["created_quarter", "created_month", "days_since_start"]

    feature_columns = base_features + engineered_features + time_features
    logger.info(f"Total features: {len(feature_columns)}")

    # 4. Train Model
    logger.info("\n[4/6] Training model...")
    # Use gradient_boosting for compatibility, or xgboost if available
    model = DealRiskScorer(model_type="gradient_boosting")
    metrics = model.train(
        X=df,
        y=df["target"],
        feature_columns=feature_columns,
        test_size=0.2
    )

    logger.info(f"Model Performance:")
    logger.info(f"  - AUC: {metrics['auc']:.4f}")
    logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  - Precision: {metrics['precision']:.4f}")
    logger.info(f"  - Recall: {metrics['recall']:.4f}")
    logger.info(f"  - F1: {metrics['f1']:.4f}")

    # Cross-validation
    cv_results = model.cross_validate(df, df["target"], cv=5)
    logger.info(f"  - CV AUC: {cv_results['mean_auc']:.4f} (+/- {cv_results['std_auc']:.4f})")

    # 5. Score all deals
    logger.info("\n[5/6] Scoring deals...")
    df = model.score_deals(df)

    # Business metrics
    business_metrics = calculate_business_metrics(df, risk_threshold=0.6)
    logger.info(f"Business Metrics:")
    logger.info(f"  - High risk deals: {business_metrics['high_risk_count']} ({business_metrics['high_risk_percentage']:.1f}%)")
    logger.info(f"  - High risk total value: ${business_metrics['high_risk_total_value']:,.0f}")
    logger.info(f"  - Expected loss: ${business_metrics['expected_loss']:,.0f}")

    # Feature importance
    logger.info("\nTop 10 Feature Importance:")
    for _, row in model.feature_importance.head(10).iterrows():
        logger.info(f"  - {row['feature']}: {row['importance']:.4f}")

    # 6. Save artifacts
    logger.info("\n[6/6] Saving artifacts...")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save processed data
    df.to_csv(output_dir / "scored_deals.csv", index=False)

    # Save model
    model.save("models/risk_scorer")

    # Save preprocessor
    preprocessor.save("models/preprocessor")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({
            "model_metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix" and k != "classification_report"},
            "cv_results": cv_results,
            "business_metrics": business_metrics,
            "data_summary": summary
        }, f, indent=2, default=str)

    # Save pipeline metrics (convert Period keys to strings)
    pipeline_metrics = calculate_pipeline_metrics(df)
    # Convert any non-string keys to strings
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k): convert_keys(v) for k, v in obj.items()}
        return obj
    pipeline_metrics = convert_keys(pipeline_metrics)
    with open(output_dir / "pipeline_metrics.json", "w") as f:
        json.dump(pipeline_metrics, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: models/risk_scorer")
    logger.info(f"Scored data saved to: {output_dir / 'scored_deals.csv'}")
    logger.info("=" * 60)

    return df, model, metrics


if __name__ == "__main__":
    df, model, metrics = main()
