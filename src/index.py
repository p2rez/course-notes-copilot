from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.chunking import CHUNKERS
from src.ingest import load_folder

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"


def build_index(strategy: str, data_dir: str = "data") -> None:
    pages = load_folder(Path(data_dir))
    chunks = CHUNKERS[strategy](pages)
    collection = f"notes_{strategy}"

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=vectors.shape[1], distance=models.Distance.COSINE
        ),
    )
    client.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(id=i, vector=vec.tolist(), payload=chunk)
            for i, (vec, chunk) in enumerate(zip(vectors, chunks))
        ],
    )
    print(f"{collection}: indexed {len(chunks)} chunks")


if __name__ == "__main__":
    for name in CHUNKERS:
        build_index(name)
