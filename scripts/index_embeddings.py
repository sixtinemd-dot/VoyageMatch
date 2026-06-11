from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_destinations
from src.database import Database
from src.preprocessing import clean_destinations
from src.rag import embedding_count, index_destinations


def main() -> None:
    database = Database()
    destinations = clean_destinations(load_destinations())
    indexed = index_destinations(database, destinations)
    print(f"Indexed {indexed} destinations.")
    print(f"Vector store rows: {embedding_count(database)}")
    print(f"Backend: {database.backend_name}")


if __name__ == "__main__":
    main()
