"""Data preprocessing for SkyGeni Sales Intelligence."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Optional
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess sales data for modeling."""

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.categorical_cols = [
            "industry", "region", "product_type",
            "lead_source", "deal_stage", "sales_rep_id"
        ]
        self.numerical_cols = ["deal_amount", "sales_cycle_days"]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessors and transform data."""
        df = df.copy()

        # Encode target
        df["target"] = (df["outcome"] == "Won").astype(int)

        # Encode categorical variables
        for col in self.categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(df[col])

        # Scale numerical features
        self.scaler = StandardScaler()
        df[["deal_amount_scaled", "sales_cycle_days_scaled"]] = self.scaler.fit_transform(
            df[self.numerical_cols]
        )

        logger.info("Preprocessing complete")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors."""
        df = df.copy()

        if "outcome" in df.columns:
            df["target"] = (df["outcome"] == "Won").astype(int)

        for col in self.categorical_cols:
            if col in self.label_encoders:
                # Handle unseen categories
                df[f"{col}_encoded"] = df[col].apply(
                    lambda x: self.label_encoders[col].transform([x])[0]
                    if x in self.label_encoders[col].classes_ else -1
                )

        if self.scaler:
            df[["deal_amount_scaled", "sales_cycle_days_scaled"]] = self.scaler.transform(
                df[self.numerical_cols]
            )

        return df

    def get_feature_columns(self) -> list:
        """Get list of feature columns for modeling."""
        encoded_cols = [f"{col}_encoded" for col in self.categorical_cols]
        scaled_cols = ["deal_amount_scaled", "sales_cycle_days_scaled"]
        return encoded_cols + scaled_cols

    def save(self, path: str) -> None:
        """Save preprocessor to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.label_encoders, f"{path}/label_encoders.joblib")
        joblib.dump(self.scaler, f"{path}/scaler.joblib")
        logger.info(f"Preprocessor saved to {path}")

    def load(self, path: str) -> None:
        """Load preprocessor from disk."""
        self.label_encoders = joblib.load(f"{path}/label_encoders.joblib")
        self.scaler = joblib.load(f"{path}/scaler.joblib")
        logger.info(f"Preprocessor loaded from {path}")


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features."""
    df = df.copy()

    # Quarter and month from created_date
    df["created_quarter"] = df["created_date"].dt.quarter
    df["created_month"] = df["created_date"].dt.month
    df["created_dayofweek"] = df["created_date"].dt.dayofweek

    # Year for trend analysis
    df["created_year"] = df["created_date"].dt.year

    # Days since start of data
    min_date = df["created_date"].min()
    df["days_since_start"] = (df["created_date"] - min_date).dt.days

    return df
