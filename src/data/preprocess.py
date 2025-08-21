import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os


def load_params(config_path="params.yaml"):
    """Load parameters from YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(data_path):
    """Load the heart attack dataset."""
    try:
        data = pd.read_csv(data_path)
        print(f"✅ Successfully loaded dataset: {data_path}")
        print(f"Dataset shape: {data.shape}")
        return data
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise


def explore_data(data):
    """Basic data exploration - same as your notebook."""
    print("\n📊 Dataset Info:")
    print(data.info())

    print("\n📊 Dataset Description:")
    print(data.describe())

    print(f"\n📊 Dataset Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

    # Check for missing values
    missing_data = data.isnull().sum()
    missing_pct = (missing_data / len(data)) * 100
    missing_df = pd.DataFrame(
        {"missing_count": missing_data, "missing_pct": missing_pct}
    )
    print("\n📊 Missing Values:")
    print(missing_df)

    # Check class distribution
    print("\n📊 Class Distribution:")
    print(data["class"].value_counts())

    return data


def preprocess_data(data, config):
    """Preprocess data - exactly as in your notebook."""
    print("\n🔄 Starting data preprocessing...")

    # Drop columns as specified in your notebook
    columns_to_drop = config["preprocessing"]["drop_columns"]
    print(f"Dropping columns: {columns_to_drop}")

    # Create a copy to avoid modifying original data
    processed_data = data.copy()

    # Drop the specified columns
    processed_data = processed_data.drop(columns=columns_to_drop)
    print(f"Data shape after dropping columns: {processed_data.shape}")

    # Separate features and target
    features = config["preprocessing"]["features"]
    target = config["preprocessing"]["target"]

    X = processed_data[features]
    y = processed_data[target]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {list(X.columns)}")

    # Verify no missing values (as seen in your notebook)
    print(f"\n✅ Missing values in features: {X.isnull().sum().sum()}")
    print(f"✅ Missing values in target: {y.isnull().sum()}")

    return X, y


def split_data(X, y, config):
    """Split data into train and test sets."""
    test_size = config["preprocessing"]["test_size"]
    random_state = config["preprocessing"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # Maintain class distribution
    )

    print(f"\n📊 Data Split Results:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training class distribution:\n{y_train.value_counts()}")
    print(f"Test class distribution:\n{y_test.value_counts()}")

    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, output_path):
    """Save processed data for model training."""
    # Combine features and target for saving
    train_data = X_train.copy()
    train_data["class"] = y_train

    test_data = X_test.copy()
    test_data["class"] = y_test

    # Create processed directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save train and test data
    train_path = output_path.replace(".csv", "_train.csv")
    test_path = output_path.replace(".csv", "_test.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"✅ Saved training data: {train_path}")
    print(f"✅ Saved test data: {test_path}")

    return train_path, test_path


def main():
    """Main preprocessing pipeline."""
    print("🚀 Starting Heart Attack Data Preprocessing Pipeline")

    # Load configuration
    config = load_params()

    # Load data
    data = load_data(config["data"]["raw_path"])

    # Explore data (same as your notebook)
    data = explore_data(data)

    # Preprocess data
    X, y = preprocess_data(data, config)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, config)

    # Save processed data
    train_path, test_path = save_processed_data(
        X_train, X_test, y_train, y_test, config["data"]["processed_path"]
    )

    print("✅ Preprocessing pipeline completed successfully!")
    return train_path, test_path


if __name__ == "__main__":
    main()
