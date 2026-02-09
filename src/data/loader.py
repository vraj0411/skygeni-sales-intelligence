"""Data loading utilities for SkyGeni Sales Intelligence."""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate sales data."""

    REQUIRED_COLUMNS = [
        "deal_id", "created_date", "closed_date", "sales_rep_id",
        "industry", "region", "product_type", "lead_source",
        "deal_stage", "deal_amount", "sales_cycle_days", "outcome"
    ]

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.data = pd.read_csv(self.data_path)
        self._validate_columns()
        self._parse_dates()

        logger.info(f"Loaded {len(self.data)} records")
        return self.data

    def _validate_columns(self) -> None:
        """Validate required columns exist."""
        missing = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _parse_dates(self) -> None:
        """Parse date columns."""
        self.data["created_date"] = pd.to_datetime(self.data["created_date"])
        self.data["closed_date"] = pd.to_datetime(self.data["closed_date"])

    def get_summary(self) -> dict:
        """Get data summary statistics."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")

        return {
            "total_deals": len(self.data),
            "date_range": {
                "start": self.data["created_date"].min().strftime("%Y-%m-%d"),
                "end": self.data["created_date"].max().strftime("%Y-%m-%d")
            },
            "outcome_distribution": self.data["outcome"].value_counts().to_dict(),
            "win_rate": (self.data["outcome"] == "Won").mean(),
            "avg_deal_amount": self.data["deal_amount"].mean(),
            "avg_sales_cycle": self.data["sales_cycle_days"].mean(),
            "unique_reps": self.data["sales_rep_id"].nunique(),
            "industries": self.data["industry"].unique().tolist(),
            "regions": self.data["region"].unique().tolist()
        }


if __name__ == "__main__":
    loader = DataLoader("data/raw/skygeni_sales_data.csv")
    df = loader.load()
    print(loader.get_summary())
