import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.database import Database
from src.recommender import INTEREST_COLUMNS


load_dotenv()

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
LOCAL_EMBEDDING_MODEL = "local-hashing-vectorizer-v1"


@dataclass
class RetrievedDestination:
    city: str
    country: str
    content: str
    metadata: dict[str, Any]
    similarity: float


@dataclass
class RagAnswer:
    answer: str
    sources: list[RetrievedDestination]
    retrieval_mode: str
    generation_mode: str


def destination_document(destination: dict[str, Any]) -> str:
    strongest = sorted(
        INTEREST_COLUMNS,
        key=lambda column: float(destination.get(column, 0)),
        reverse=True,
    )[:3]
    interest_summary = ", ".join(
        f"{interest} {float(destination.get(interest, 0)):.0f}/10"
        for interest in strongest
    )
    return (
        f"{destination['city']}, {destination['country']} is in "
        f"{str(destination['region']).replace('_', ' ')}. "
        f"{destination.get('description', '')} "
        f"Typical annual temperature: {float(destination['avg_temp_c']):.1f} C. "
        f"Estimated base daily cost: {float(destination['cost_per_day_usd']):.0f} USD. "
        f"Strongest travel qualities: {interest_summary}. "
        f"Safety rating: {float(destination['safety']):.0f}/10. "
        f"Popularity rating: {float(destination['popularity']):.0f}/10."
    )


def _metadata(destination: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "city",
        "country",
        "region",
        "cost_per_day_usd",
        "avg_temp_c",
        "beach",
        "culture",
        "nature",
        "nightlife",
        "food",
        "adventure",
        "relaxation",
        "popularity",
        "safety",
    ]
    return {
        field: (
            destination[field].item()
            if hasattr(destination.get(field), "item")
            else destination.get(field)
        )
        for field in fields
    }


def openai_is_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def ensure_vector_store(database: Database) -> None:
    with database.connection() as connection:
        cursor = connection.cursor()
        if database.is_postgres:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS destination_embeddings (
                    city TEXT NOT NULL,
                    country TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding vector({DEFAULT_EMBEDDING_DIMENSIONS}) NOT NULL,
                    embedding_model TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (city, country)
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS destination_embeddings (
                    city TEXT NOT NULL,
                    country TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (city, country)
                )
                """
            )


def embedding_count(database: Database) -> int:
    ensure_vector_store(database)
    with database.connection() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM destination_embeddings")
        row = cursor.fetchone()
    return int(row["count"])


def _create_embeddings(texts: list[str]) -> list[list[float]]:
    if not openai_is_configured():
        raise RuntimeError("OPENAI_API_KEY is required to create vector embeddings.")

    from openai import OpenAI

    response = OpenAI(max_retries=0).embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        input=texts,
        dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        encoding_format="float",
    )
    return [item.embedding for item in response.data]


def _create_local_embeddings(texts: list[str]) -> list[list[float]]:
    vectorizer = HashingVectorizer(
        n_features=DEFAULT_EMBEDDING_DIMENSIONS,
        alternate_sign=False,
        norm="l2",
        stop_words="english",
        ngram_range=(1, 2),
    )
    return vectorizer.transform(texts).toarray().astype(float).tolist()


def _stored_embedding_model(database: Database) -> str | None:
    ensure_vector_store(database)
    with database.connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT embedding_model
            FROM destination_embeddings
            LIMIT 1
            """
        )
        row = cursor.fetchone()
    return str(row["embedding_model"]) if row else None


def embedding_backend_label(database: Database) -> str:
    model = _stored_embedding_model(database)
    if model == LOCAL_EMBEDDING_MODEL:
        return "local text vectors"
    if model:
        return f"OpenAI embeddings ({model})"
    return "not indexed"


def _clear_vector_store(database: Database) -> None:
    with database.connection() as connection:
        connection.execute("DELETE FROM destination_embeddings")


def index_destinations(
    database: Database,
    destinations: pd.DataFrame,
    batch_size: int = 64,
) -> int:
    ensure_vector_store(database)
    records = destinations.to_dict(orient="records")
    contents = [destination_document(record) for record in records]
    model = LOCAL_EMBEDDING_MODEL
    embedding_batches: list[list[list[float]]] = []

    if openai_is_configured():
        try:
            for start in range(0, len(contents), batch_size):
                embedding_batches.append(
                    _create_embeddings(contents[start : start + batch_size])
                )
            model = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        except Exception:
            embedding_batches = []

    if not embedding_batches:
        for start in range(0, len(contents), batch_size):
            embedding_batches.append(
                _create_local_embeddings(contents[start : start + batch_size])
            )

    _clear_vector_store(database)
    indexed = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        batch_contents = contents[start : start + batch_size]
        embeddings = embedding_batches[start // batch_size]
        with database.connection() as connection:
            cursor = connection.cursor()
            for record, content, embedding in zip(
                batch,
                batch_contents,
                embeddings,
            ):
                metadata_json = json.dumps(_metadata(record), default=str)
                if database.is_postgres:
                    vector_literal = "[" + ",".join(map(str, embedding)) + "]"
                    cursor.execute(
                        """
                        INSERT INTO destination_embeddings (
                            city, country, content, metadata_json, embedding,
                            embedding_model, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                        ON CONFLICT(city, country) DO UPDATE SET
                            content = excluded.content,
                            metadata_json = excluded.metadata_json,
                            embedding = excluded.embedding,
                            embedding_model = excluded.embedding_model,
                            updated_at = excluded.updated_at
                        """,
                        (
                            record["city"],
                            record["country"],
                            content,
                            metadata_json,
                            vector_literal,
                            model,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO destination_embeddings (
                            city, country, content, metadata_json, embedding_json,
                            embedding_model, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(city, country) DO UPDATE SET
                            content = excluded.content,
                            metadata_json = excluded.metadata_json,
                            embedding_json = excluded.embedding_json,
                            embedding_model = excluded.embedding_model,
                            updated_at = excluded.updated_at
                        """,
                        (
                            record["city"],
                            record["country"],
                            content,
                            metadata_json,
                            json.dumps(embedding),
                            model,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
        indexed += len(batch)
    return indexed


def _vector_retrieve(
    database: Database,
    question: str,
    top_k: int,
) -> list[RetrievedDestination]:
    model = _stored_embedding_model(database)
    if model == LOCAL_EMBEDDING_MODEL:
        query_embedding = _create_local_embeddings([question])[0]
    else:
        query_embedding = _create_embeddings([question])[0]
    with database.connection() as connection:
        cursor = connection.cursor()
        if database.is_postgres:
            vector_literal = "[" + ",".join(map(str, query_embedding)) + "]"
            cursor.execute(
                """
                SELECT city, country, content, metadata_json,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM destination_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector_literal, vector_literal, top_k),
            )
        else:
            cursor.execute(
                """
                SELECT city, country, content, metadata_json, embedding_json
                FROM destination_embeddings
                """
            )
        rows = cursor.fetchall()

    if not database.is_postgres:
        query = np.asarray(query_embedding, dtype=float).reshape(1, -1)
        scored_rows = []
        for row in rows:
            embedding = np.asarray(json.loads(row["embedding_json"]), dtype=float)
            similarity = float(cosine_similarity(query, embedding.reshape(1, -1))[0, 0])
            scored_rows.append((similarity, row))
        rows = [
            {**dict(row), "similarity": similarity}
            for similarity, row in sorted(
                scored_rows,
                key=lambda item: item[0],
                reverse=True,
            )[:top_k]
        ]

    return [
        RetrievedDestination(
            city=str(row["city"]),
            country=str(row["country"]),
            content=str(row["content"]),
            metadata=json.loads(row["metadata_json"]),
            similarity=float(row["similarity"]),
        )
        for row in rows
    ]


def _local_retrieve(
    destinations: pd.DataFrame,
    question: str,
    top_k: int,
) -> list[RetrievedDestination]:
    records = destinations.to_dict(orient="records")
    contents = [destination_document(record) for record in records]
    matrix = TfidfVectorizer(stop_words="english").fit_transform(
        contents + [question]
    )
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    indices = similarities.argsort()[::-1][:top_k]
    return [
        RetrievedDestination(
            city=str(records[index]["city"]),
            country=str(records[index]["country"]),
            content=contents[index],
            metadata=_metadata(records[index]),
            similarity=float(similarities[index]),
        )
        for index in indices
    ]


def retrieve_destinations(
    database: Database,
    destinations: pd.DataFrame,
    question: str,
    top_k: int = 5,
) -> tuple[list[RetrievedDestination], str]:
    if embedding_count(database) > 0:
        model = _stored_embedding_model(database)
        backend = "Neon pgvector" if database.is_postgres else "local vector store"
        embedding_source = (
            "local text vectors"
            if model == LOCAL_EMBEDDING_MODEL
            else "OpenAI embeddings"
        )
        return _vector_retrieve(
            database,
            question,
            top_k,
        ), f"{embedding_source} + {backend}"
    return _local_retrieve(destinations, question, top_k), "local TF-IDF fallback"


def _grounded_fallback_answer(
    question: str,
    sources: list[RetrievedDestination],
) -> str:
    lines = [
        "Based on the most relevant destinations in the VoyageMatch dataset:",
        "",
    ]
    for source in sources[:3]:
        metadata = source.metadata
        lines.append(
            f"- **{source.city}, {source.country}**: "
            f"{metadata['avg_temp_c']:.1f} C typical temperature, "
            f"about ${metadata['cost_per_day_usd']:.0f}/day, with "
            f"culture {metadata['culture']:.0f}/10, nature "
            f"{metadata['nature']:.0f}/10, and beach "
            f"{metadata['beach']:.0f}/10."
        )
    lines.extend(
        [
            "",
            "This is a retrieval-only answer because OpenAI generation is not configured. "
            f"The search interpreted your question as: “{question}”",
        ]
    )
    return "\n".join(lines)


def answer_travel_question(
    database: Database,
    destinations: pd.DataFrame,
    question: str,
    top_k: int = 5,
) -> RagAnswer:
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Enter a travel question.")

    sources, retrieval_mode = retrieve_destinations(
        database,
        destinations,
        clean_question,
        top_k,
    )
    if not openai_is_configured():
        return RagAnswer(
            answer=_grounded_fallback_answer(clean_question, sources),
            sources=sources,
            retrieval_mode=retrieval_mode,
            generation_mode="grounded template fallback",
        )

    from openai import OpenAI

    context = "\n\n".join(
        f"[{index}] {source.content}"
        for index, source in enumerate(sources, start=1)
    )
    prompt = f"""
You are the VoyageMatch travel assistant.

Answer the user's question using only the retrieved destination context below.
Do not invent prices, ratings, weather, attractions, or facts that are not in
the context. Compare destinations when useful. Cite supporting destinations
with bracket numbers such as [1] or [2]. If the context is insufficient, say so.

User question:
{clean_question}

Retrieved context:
{context}

Return a concise, helpful answer in Markdown.
"""
    try:
        response = OpenAI(max_retries=0).responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
            input=prompt,
        )
        answer = response.output_text
        generation_mode = "OpenAI transformer inference"
    except Exception:
        answer = _grounded_fallback_answer(clean_question, sources)
        generation_mode = "grounded template fallback"

    return RagAnswer(
        answer=answer,
        sources=sources,
        retrieval_mode=retrieval_mode,
        generation_mode=generation_mode,
    )
