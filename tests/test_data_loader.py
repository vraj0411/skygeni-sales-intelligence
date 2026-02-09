"""Tests for data loading module."""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    @pytest.fixture
    def sample_data_path(self, tmp_path):
        """Create sample data for testing."""
        data = pd.DataFrame({
            "deal_id": ["D001", "D002", "D003"],
            "created_date": ["2024-01-01", "2024-01-15", "2024-02-01"],
            "closed_date": ["2024-01-15", "2024-02-01", "2024-02-15"],
            "sales_rep_id": ["rep_1", "rep_2", "rep_1"],
            "industry": ["SaaS", "FinTech", "SaaS"],
            "region": ["North America", "Europe", "APAC"],
            "product_type": ["Enterprise", "Core", "Pro"],
            "lead_source": ["Inbound", "Outbound", "Referral"],
            "deal_stage": ["Closed", "Negotiation", "Proposal"],
            "deal_amount": [10000, 5000, 7500],
            "sales_cycle_days": [14, 17, 14],
            "outcome": ["Won", "Lost", "Won"]
        })
        file_path = tmp_path / "test_data.csv"
        data.to_csv(file_path, index=False)
        return str(file_path)

    def test_load_data(self, sample_data_path):
        """Test data loading."""
        loader = DataLoader(sample_data_path)
        df = loader.load()

        assert len(df) == 3
        assert "deal_id" in df.columns
        assert df["created_date"].dtype == "datetime64[ns]"

    def test_get_summary(self, sample_data_path):
        """Test summary generation."""
        loader = DataLoader(sample_data_path)
        loader.load()
        summary = loader.get_summary()

        assert summary["total_deals"] == 3
        assert "win_rate" in summary
        assert summary["unique_reps"] == 2

    def test_missing_file(self):
        """Test handling of missing file."""
        loader = DataLoader("nonexistent.csv")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_missing_columns(self, tmp_path):
        """Test handling of missing required columns."""
        data = pd.DataFrame({"deal_id": ["D001"], "amount": [1000]})
        file_path = tmp_path / "incomplete.csv"
        data.to_csv(file_path, index=False)

        loader = DataLoader(str(file_path))

        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load()


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe."""
        return pd.DataFrame({
            "deal_id": ["D001", "D002"],
            "industry": ["SaaS", "FinTech"],
            "region": ["NA", "EU"],
            "product_type": ["Core", "Pro"],
            "lead_source": ["Inbound", "Outbound"],
            "deal_stage": ["Closed", "Proposal"],
            "sales_rep_id": ["rep_1", "rep_2"],
            "deal_amount": [10000, 5000],
            "sales_cycle_days": [30, 45],
            "outcome": ["Won", "Lost"]
        })

    def test_fit_transform(self, sample_df):
        """Test preprocessing fit and transform."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(sample_df)

        assert "target" in result.columns
        assert "industry_encoded" in result.columns
        assert "deal_amount_scaled" in result.columns

    def test_get_feature_columns(self, sample_df):
        """Test feature column list."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_df)
        features = preprocessor.get_feature_columns()

        assert len(features) > 0
        assert all("encoded" in f or "scaled" in f for f in features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
