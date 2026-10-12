from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.index import MODEL_NAME, QDRANT_URL

_model = SentenceTransformer(MODEL_NAME)
_client = QdrantClient(url=QDRANT_URL)


def search(query: str, strategy: str = "page", k: int = 3, course: str | None = None) -> list[dict]:
    vector = _model.encode(query, normalize_embeddings=True).tolist()
    query_filter = None
    if course:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="course", match=models.MatchValue(value=course))]
        )
    result = _client.query_points(
        collection_name=f"notes_{strategy}",
        query=vector,
        limit=k,
        query_filter=query_filter,
    )
    return [{"score": p.score, **p.payload} for p in result.points]
