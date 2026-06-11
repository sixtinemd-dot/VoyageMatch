import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset_builder import RAW_DIR, build_processed_dataset, discover_raw_csv


KAGGLE_DATASET = "furkanima/worldwide-travel-cities-ratings-and-climate"


def download_kaggle_dataset() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required only when no raw dataset is available. "
            "Install the project requirements or add the source CSV to data/raw."
        ) from exc

    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_files = list(downloaded_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset: {downloaded_path}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source_csv = csv_files[0]
    target_csv = RAW_DIR / source_csv.name
    target_csv.write_bytes(source_csv.read_bytes())
    return target_csv


def main() -> None:
    raw_csv = discover_raw_csv()
    if raw_csv is None:
        print("No Kaggle CSV found in data/raw. Downloading with kagglehub...")
        raw_csv = download_kaggle_dataset()
    else:
        print(f"Using existing raw dataset: {raw_csv}")

    processed = build_processed_dataset(raw_csv)
    print(f"Saved combined dataset with {len(processed)} destinations.")
    print("Output: data/processed/destinations.csv")


if __name__ == "__main__":
    main()
