import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from dotenv import load_dotenv


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE_PATH = ROOT_DIR / "data" / "voyagematch.db"
PASSWORD_ITERATIONS = 600_000


@dataclass
class User:
    id: int
    email: str
    display_name: str
    created_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(candidate.hex(), digest_hex)
    except (TypeError, ValueError):
        return False


class Database:
    """Persistence service for local SQLite and hosted Neon PostgreSQL."""

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            f"sqlite:///{DEFAULT_SQLITE_PATH}",
        )
        self.is_postgres = self.database_url.startswith(
            ("postgresql://", "postgres://")
        )
        self.placeholder = "%s" if self.is_postgres else "?"
        self._initialized = False

    @property
    def backend_name(self) -> str:
        return "Neon PostgreSQL" if self.is_postgres else "local SQLite"

    @contextmanager
    def connection(self) -> Iterator[Any]:
        if self.is_postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL support requires psycopg. "
                    "Install the packages in requirements.txt."
                ) from exc

            connection = psycopg.connect(
                self.database_url,
                row_factory=dict_row,
                connect_timeout=5,
            )
        else:
            parsed = urlparse(self.database_url)
            database_path = Path(parsed.path)
            database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(database_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        if self._initialized:
            return

        primary_key = (
            "BIGSERIAL PRIMARY KEY"
            if self.is_postgres
            else "INTEGER PRIMARY KEY AUTOINCREMENT"
        )
        statements = [
            f"""
            CREATE TABLE IF NOT EXISTS users (
                id {primary_key},
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                preferences_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS favorites (
                id {primary_key},
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                city TEXT NOT NULL,
                country TEXT NOT NULL,
                destination_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, city, country)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS recommendation_history (
                id {primary_key},
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                preferences_json TEXT NOT NULL,
                recommendations_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS saved_itineraries (
                id {primary_key},
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                city TEXT NOT NULL,
                country TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                itinerary TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
        ]
        with self.connection() as connection:
            cursor = connection.cursor()
            for statement in statements:
                cursor.execute(statement)
        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def register_user(
        self,
        email: str,
        password: str,
        display_name: str,
    ) -> User:
        self._ensure_initialized()
        normalized_email = email.strip().lower()
        normalized_name = display_name.strip()
        if "@" not in normalized_email or "." not in normalized_email.rsplit("@", 1)[-1]:
            raise ValueError("Enter a valid email address.")
        if len(password) < 8:
            raise ValueError("Password must contain at least 8 characters.")
        if not normalized_name:
            raise ValueError("Display name is required.")

        try:
            with self.connection() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    f"""
                    INSERT INTO users (email, display_name, password_hash, created_at)
                    VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})
                    """,
                    (
                        normalized_email,
                        normalized_name,
                        _hash_password(password),
                        _utc_now(),
                    ),
                )
        except Exception as exc:
            if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                raise ValueError("An account with this email already exists.") from exc
            raise

        user = self.get_user_by_email(normalized_email)
        if user is None:
            raise RuntimeError("The account was created but could not be loaded.")
        return user

    def authenticate(self, email: str, password: str) -> User | None:
        self._ensure_initialized()
        normalized_email = email.strip().lower()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT id, email, display_name, password_hash, created_at
                FROM users
                WHERE email = {self.placeholder}
                """,
                (normalized_email,),
            )
            row = cursor.fetchone()
        if not row or not _verify_password(password, row["password_hash"]):
            return None
        return User(
            id=int(row["id"]),
            email=str(row["email"]),
            display_name=str(row["display_name"]),
            created_at=str(row["created_at"]),
        )

    def get_user_by_email(self, email: str) -> User | None:
        self._ensure_initialized()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT id, email, display_name, created_at
                FROM users
                WHERE email = {self.placeholder}
                """,
                (email.strip().lower(),),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return User(
            id=int(row["id"]),
            email=str(row["email"]),
            display_name=str(row["display_name"]),
            created_at=str(row["created_at"]),
        )

    def save_preferences(self, user_id: int, preferences: dict[str, Any]) -> None:
        self._ensure_initialized()
        payload = json.dumps(preferences, default=str)
        now = _utc_now()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                INSERT INTO user_preferences (user_id, preferences_json, updated_at)
                VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder})
                ON CONFLICT(user_id) DO UPDATE SET
                    preferences_json = excluded.preferences_json,
                    updated_at = excluded.updated_at
                """,
                (user_id, payload, now),
            )

    def load_preferences(self, user_id: int) -> dict[str, Any] | None:
        self._ensure_initialized()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT preferences_json
                FROM user_preferences
                WHERE user_id = {self.placeholder}
                """,
                (user_id,),
            )
            row = cursor.fetchone()
        return json.loads(row["preferences_json"]) if row else None

    def add_favorite(
        self,
        user_id: int,
        destination: dict[str, Any],
    ) -> bool:
        self._ensure_initialized()
        city = str(destination["city"])
        country = str(destination["country"])
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                INSERT INTO favorites (
                    user_id, city, country, destination_json, created_at
                )
                VALUES (
                    {self.placeholder}, {self.placeholder}, {self.placeholder},
                    {self.placeholder}, {self.placeholder}
                )
                ON CONFLICT(user_id, city, country) DO NOTHING
                """,
                (
                    user_id,
                    city,
                    country,
                    json.dumps(destination, default=str),
                    _utc_now(),
                ),
            )
            return cursor.rowcount > 0

    def remove_favorite(self, user_id: int, city: str, country: str) -> None:
        self._ensure_initialized()
        with self.connection() as connection:
            connection.execute(
                f"""
                DELETE FROM favorites
                WHERE user_id = {self.placeholder}
                  AND city = {self.placeholder}
                  AND country = {self.placeholder}
                """,
                (user_id, city, country),
            )

    def is_favorite(self, user_id: int, city: str, country: str) -> bool:
        self._ensure_initialized()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT 1 FROM favorites
                WHERE user_id = {self.placeholder}
                  AND city = {self.placeholder}
                  AND country = {self.placeholder}
                """,
                (user_id, city, country),
            )
            return cursor.fetchone() is not None

    def list_favorites(self, user_id: int) -> list[dict[str, Any]]:
        self._ensure_initialized()
        return self._fetch_all(
            f"""
            SELECT id, city, country, destination_json, created_at
            FROM favorites
            WHERE user_id = {self.placeholder}
            ORDER BY created_at DESC
            """,
            (user_id,),
        )

    def save_recommendation_history(
        self,
        user_id: int,
        preferences: dict[str, Any],
        recommendations: list[dict[str, Any]],
    ) -> None:
        self._ensure_initialized()
        with self.connection() as connection:
            connection.execute(
                f"""
                INSERT INTO recommendation_history (
                    user_id, preferences_json, recommendations_json, created_at
                )
                VALUES (
                    {self.placeholder}, {self.placeholder},
                    {self.placeholder}, {self.placeholder}
                )
                """,
                (
                    user_id,
                    json.dumps(preferences, default=str),
                    json.dumps(recommendations, default=str),
                    _utc_now(),
                ),
            )

    def list_recommendation_history(
        self,
        user_id: int,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        self._ensure_initialized()
        return self._fetch_all(
            f"""
            SELECT id, preferences_json, recommendations_json, created_at
            FROM recommendation_history
            WHERE user_id = {self.placeholder}
            ORDER BY created_at DESC
            LIMIT {self.placeholder}
            """,
            (user_id, limit),
        )

    def save_itinerary(
        self,
        user_id: int,
        city: str,
        country: str,
        start_date: date,
        end_date: date,
        itinerary: str,
    ) -> None:
        self._ensure_initialized()
        with self.connection() as connection:
            connection.execute(
                f"""
                INSERT INTO saved_itineraries (
                    user_id, city, country, start_date, end_date,
                    itinerary, created_at
                )
                VALUES (
                    {self.placeholder}, {self.placeholder}, {self.placeholder},
                    {self.placeholder}, {self.placeholder}, {self.placeholder},
                    {self.placeholder}
                )
                """,
                (
                    user_id,
                    city,
                    country,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    itinerary,
                    _utc_now(),
                ),
            )

    def list_itineraries(self, user_id: int) -> list[dict[str, Any]]:
        self._ensure_initialized()
        return self._fetch_all(
            f"""
            SELECT id, city, country, start_date, end_date, itinerary, created_at
            FROM saved_itineraries
            WHERE user_id = {self.placeholder}
            ORDER BY created_at DESC
            """,
            (user_id,),
        )

    def delete_itinerary(self, user_id: int, itinerary_id: int) -> None:
        self._ensure_initialized()
        with self.connection() as connection:
            connection.execute(
                f"""
                DELETE FROM saved_itineraries
                WHERE id = {self.placeholder} AND user_id = {self.placeholder}
                """,
                (itinerary_id, user_id),
            )

    def _fetch_all(
        self,
        query: str,
        parameters: tuple[Any, ...],
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(query, parameters)
            return [dict(row) for row in cursor.fetchall()]


def user_to_dict(user: User) -> dict[str, Any]:
    return asdict(user)


def user_from_dict(data: dict[str, Any]) -> User:
    return User(**data)
