from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = ROOT_DIR / "data" / "raw" / "sample_destinations.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "destinations.csv"


def load_destinations(path: Path | None = None) -> pd.DataFrame:
    """Load processed destination data when available, otherwise sample data."""
    selected_path = path or (
        PROCESSED_DATA_PATH if PROCESSED_DATA_PATH.exists() else SAMPLE_DATA_PATH
    )
    return pd.read_csv(selected_path)
