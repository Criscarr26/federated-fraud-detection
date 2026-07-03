"""Data loading, cleaning, splitting and preprocessing utilities.

Functions:
- load_raw_data(path)
- basic_cleaning(df, label_col="label")
- train_test_split_global(df)
- scale_features(X_train, X_test)
- split_train_among_clients(X_train, y_train, n_clients=3)
"""
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV data from path.

    Args:
        path: Path to CSV file.

    Returns:
        DataFrame with the raw data.
    """
    df = pd.read_csv(path)
    return df


def basic_cleaning(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Basic cleaning: drop NA, one-hot encode categorical cols and ensure label exists.

    Args:
        df: Raw dataframe.
        label_col: Name of label column.

    Returns:
        Cleaned dataframe ready for train/test split.
    """
    df = df.copy()
    # Drop rows with NA in any column
    df = df.dropna().reset_index(drop=True)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")

    # Identify categorical columns (object or category)
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != label_col]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def train_test_split_global(df: pd.DataFrame, label_col: str = "label", test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into global train and test.

    Args:
        df: Cleaned dataframe.
        label_col: Label column name.
        test_size: Fraction for test set.
        random_state: RNG seed.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features using StandardScaler.

    Args:
        X_train: Training features.
        X_test: Test features.

    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def split_train_among_clients(X_train: np.ndarray, y_train: np.ndarray, n_clients: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split training set among n_clients (roughly equal).

    Args:
        X_train: Numpy array of training features.
        y_train: Numpy array of training labels.
        n_clients: Number of clients to split among.

    Returns:
        List of (X_client, y_client) tuples.
    """
    if isinstance(X_train, np.ndarray) is False:
        X_train = np.asarray(X_train)
    if isinstance(y_train, np.ndarray) is False:
        y_train = np.asarray(y_train)

    indices = np.arange(len(X_train))
    # Shuffle before splitting
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    chunks = np.array_split(indices, n_clients)
    splits = []
    for c in chunks:
        splits.append((X_train[c], y_train[c]))
    return splits


def data_quality_report(df: pd.DataFrame, label_col: str = "label") -> Dict[str, Any]:
    """Return a data quality report with nulls, types and basic malformed counts.

    "Malformed" is conservatively detected as non-numeric values in columns
    that pandas infers as numeric (after coercion check), and values not in
    {0,1} for the label expected to be binary.

    Returns a dict with counts and simple summaries.
    """
    report: Dict[str, Any] = {}
    report["n_rows"] = int(len(df))
    report["n_cols"] = int(len(df.columns))
    # Null counts per column
    report["null_counts"] = df.isnull().sum().to_dict()

    # Malformed: for each column attempt to coerce to numeric if dtype is object
    malformed: Dict[str, int] = {}
    for col in df.columns:
        if col == label_col:
            # check binary values
            uniques = pd.Series(df[col].unique()).dropna()
            malformed[col] = int(~uniques.isin([0, 1]).all()) if len(uniques) > 0 else 0
            continue
        # try to coerce
        coerced = pd.to_numeric(df[col], errors="coerce")
        # count non-numeric cells where original was non-null
        non_numeric = int(((coerced.isnull()) & (~df[col].isnull())).sum())
        malformed[col] = non_numeric
    report["malformed_counts"] = malformed

    # Feature count after one-hot encoding (approx):
    # If there are object dtypes we simulate get_dummies to estimate final feature count
    obj_cols = [c for c in df.columns if df[c].dtype == "object" and c != label_col]
    if obj_cols:
        df_dum = pd.get_dummies(df.drop(columns=[label_col]), columns=obj_cols, drop_first=True)
        report["n_features_after_encoding"] = int(df_dum.shape[1])
    else:
        report["n_features_after_encoding"] = int(len(df.columns) - 1)

    return report


def validate_dataset(df: pd.DataFrame, label_col: str = "label", n_clients: int = 3, min_per_client: int = 5000, min_features: int = 6) -> Dict[str, Any]:
    """Validate that dataset meets requirements.

    Requirements checked:
    - label_col exists and is binary (0/1)
    - after basic cleaning and one-hot encoding, there are at least min_features input features
    - after splitting into n_clients, each client has at least min_per_client rows

    Returns a dict: {'ok': bool, 'errors': List[str], 'report': data_quality_report}
    """
    errors: List[str] = []
    report = data_quality_report(df, label_col=label_col)

    # Check label exists
    if label_col not in df.columns:
        errors.append(f"Label column '{label_col}' not found")
        return {"ok": False, "errors": errors, "report": report}

    # Check label binary
    unique_labels = pd.Series(df[label_col].dropna().unique())
    if not set(unique_labels).issubset({0, 1}):
        errors.append("Label values are not binary (0/1)")

    # Check features after encoding
    obj_cols = [c for c in df.columns if df[c].dtype == "object" and c != label_col]
    if obj_cols:
        df_enc = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    else:
        df_enc = df.copy()
    n_features = df_enc.shape[1] - 1 if label_col in df_enc.columns else df_enc.shape[1]
    if n_features < min_features:
        errors.append(f"Not enough features after encoding: {n_features} < {min_features}")

    # Check size per client (use cleaned + encoded X)
    X = df_enc.drop(columns=[label_col])
    y = df_enc[label_col]
    splits = split_train_among_clients(X.values, y.values, n_clients=n_clients)
    small_clients = [i for i, (Xc, yc) in enumerate(splits) if len(Xc) < min_per_client]
    if small_clients:
        errors.append(f"Clients with insufficient rows (min {min_per_client}): {small_clients}")

    ok = len(errors) == 0
    return {"ok": ok, "errors": errors, "report": report}
