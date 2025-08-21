import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data.preprocess import (
    load_data,
    explore_data,
    preprocess_data,
    split_data,
    save_processed_data,
    load_params,
)


@pytest.fixture
# "This above function provides data or setup that my tests need. Run this function and give its results to any test that requests it."
def sample_heart_attack_data():
    """Create sample heart attack data for testing."""
    data = {
        "age": [64, 21, 55, 64, 55],
        "gender": [1, 1, 1, 1, 1],
        "impluse": [66, 94, 64, 70, 64],  # Will be dropped
        "pressurehight": [160, 98, 160, 120, 112],
        "pressurelow": [83, 46, 77, 55, 65],
        "glucose": [160.0, 296.0, 270.0, 270.0, 300.0],
        "kcm": [1.80, 6.75, 1.99, 13.87, 1.08],
        "troponin": [0.012, 1.060, 0.003, 0.122, 0.003],
        "class": ["negative", "positive", "negative", "positive", "negative"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "data": {"raw_path": "test_data.csv", "processed_path": "test_processed.csv"},
        "preprocessing": {
            "test_size": 0.4,
            "random_state": 42,
            "features": [
                "age",
                "gender",
                "pressurehight",
                "pressurelow",
                "glucose",
                "kcm",
                "troponin",
            ],
            "target": "class",
            "drop_columns": ["impluse"],
        },
    }


class TestDataLoading:
    def test_load_data_success(self, sample_heart_attack_data):
        """Test successful data loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_heart_attack_data.to_csv(f.name, index=False)

            loaded_data = load_data(f.name)

            assert loaded_data.shape == sample_heart_attack_data.shape
            assert list(loaded_data.columns) == list(sample_heart_attack_data.columns)

        os.unlink(f.name)

    def test_load_data_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(Exception):
            load_data("non_existent_file.csv")


class TestDataExploration:
    def test_explore_data(self, sample_heart_attack_data):
        """Test data exploration function."""
        result = explore_data(sample_heart_attack_data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_heart_attack_data.shape


class TestPreprocessing:
    def test_preprocess_data(self, sample_heart_attack_data, test_config):
        """Test data preprocessing."""
        X, y = preprocess_data(sample_heart_attack_data, test_config)

        # Check that 'impluse' column is dropped
        assert "impluse" not in X.columns

        # Check feature dimensions
        expected_features = test_config["preprocessing"]["features"]
        assert list(X.columns) == expected_features
        assert len(X.columns) == len(expected_features)

        # Check target
        assert len(y) == len(sample_heart_attack_data)
        assert y.name == test_config["preprocessing"]["target"]

    def test_split_data_stratification(self, sample_heart_attack_data, test_config):
        """Test that data splitting maintains class distribution."""
        X, y = preprocess_data(sample_heart_attack_data, test_config)
        X_train, X_test, y_train, y_test = split_data(X, y, test_config)

        # Check that we have train and test sets
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0

        # Check that total samples are preserved
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

    def test_save_processed_data(self, sample_heart_attack_data, test_config):
        """Test saving processed data."""
        X, y = preprocess_data(sample_heart_attack_data, test_config)
        X_train, X_test, y_train, y_test = split_data(X, y, test_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "processed_data.csv")

            train_path, test_path = save_processed_data(
                X_train, X_test, y_train, y_test, output_path
            )

            # Check that files are created
            assert os.path.exists(train_path)
            assert os.path.exists(test_path)

            # Check that data can be loaded back
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            assert len(train_data) == len(X_train)
            assert len(test_data) == len(X_test)
            assert "class" in train_data.columns
            assert "class" in test_data.columns


class TestDataQuality:
    def test_no_missing_values(self, sample_heart_attack_data, test_config):
        """Test that processed data has no missing values."""
        X, y = preprocess_data(sample_heart_attack_data, test_config)

        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0

    def test_feature_types(self, sample_heart_attack_data, test_config):
        """Test that features have correct data types."""
        X, y = preprocess_data(sample_heart_attack_data, test_config)

        # All features should be numeric
        for col in X.columns:
            assert pd.api.types.is_numeric_dtype(X[col])

        # Target should be string/object
        assert pd.api.types.is_object_dtype(y)

    def test_class_distribution(self, sample_heart_attack_data):
        """Test class distribution in sample data."""
        class_counts = sample_heart_attack_data["class"].value_counts()

        assert "positive" in class_counts.index
        assert "negative" in class_counts.index
        assert class_counts.sum() == len(sample_heart_attack_data)
