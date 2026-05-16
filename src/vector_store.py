import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path


CHROMA_PATH = "./data/chroma_db"


def get_collection(collection_name: str = "papers"):
    """
    Initialize ChromaDB client and return (or create) a collection.
    Uses the default sentence-transformers embedding model locally — no API key needed.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Uses all-MiniLM-L6-v2 by default — fast, good quality, runs locally
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_chunks(chunks: list[dict], collection_name: str = "papers"):
    """
    Add parsed paper chunks to the vector store.
    Each chunk gets embedded and stored with its metadata.
    """
    collection = get_collection(collection_name)

    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "paper_id": c["paper_id"],
            "section": c["section"],
            "source": c["source"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]
    ids = [
        f"{c['paper_id']}_{c['section']}_{c['chunk_index']}"
        for c in chunks
    ]

    # ChromaDB handles deduplication by ID — safe to re-add
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Added {len(chunks)} chunks to collection '{collection_name}'")


def query(
    question: str,
    n_results: int = 5,
    paper_ids: list[str] | None = None,
    collection_name: str = "papers",
) -> list[dict]:
    """
    Semantic search over stored chunks.
    Optionally filter by specific paper_ids.
    Returns list of dicts with text + metadata + distance score.
    """
    collection = get_collection(collection_name)

    where_filter = None
    if paper_ids:
        where_filter = {"paper_id": {"$in": paper_ids}}

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "paper_id": meta["paper_id"],
            "section": meta["section"],
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "relevance_score": round(1 - dist, 3),  # cosine similarity
        })

    return chunks


def list_papers(collection_name: str = "papers") -> list[str]:
    """Return unique paper_ids currently in the collection."""
    collection = get_collection(collection_name)
    results = collection.get(include=["metadatas"])
    paper_ids = list({m["paper_id"] for m in results["metadatas"]})
    return sorted(paper_ids)


if __name__ == "__main__":
    # Quick test
    test_chunks = [
        {
            "paper_id": "test_paper",
            "section": "abstract",
            "chunk_index": 0,
            "text": "The hippocampus plays a critical role in memory consolidation and spatial navigation.",
            "source": "test.pdf",
        }
    ]
    add_chunks(test_chunks)
    results = query("What does the hippocampus do?")
    for r in results:
        print(f"[{r['relevance_score']}] {r['source']} ({r['section']}): {r['text'][:80]}...")
