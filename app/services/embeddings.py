from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# load the embedding model once at startup
# all-MiniLM-L6-v2 is small, fast, and good quality
# downloads automatically on first run (~90MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

# create ChromaDB client — stores data locally in ./chroma_db folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# get or create a collection — like a table in a regular database
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # correct
)

def embed_text(text: str) -> list[float]:
    """Convert text to embedding vector"""
    embedding = model.encode(text)
    return embedding.tolist()


def add_document(doc_id: str, text: str, metadata: dict = {}):
    """Add a document to the vector store"""
    embedding = embed_text(text)
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata]
    )
    return {"id": doc_id, "status": "added"}


def search_documents(query: str, n_results: int = 3) -> list[dict]:
    """Find most similar documents to a query"""
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # format results cleanly
    documents = []
    for i in range(len(results['ids'][0])):
        documents.append({
            "id": results['ids'][0][i],
            "text": results['documents'][0][i],
            "similarity": round(1 - results['distances'][0][i], 4),
            "metadata": results['metadatas'][0][i]
        })

    return documents


def get_collection_count() -> int:
    """How many documents are in the collection"""
    return collection.count()

def add_document_chunks(chunks: list, doc_id_prefix: str) -> dict:
    """
    Add multiple chunks from a document to ChromaDB.
    Each chunk gets its own vector and metadata.
    """
    if not chunks:
        return {"status": "no chunks to add", "count": 0}

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id_prefix}_chunk_{i}"
        embedding = embed_text(chunk["text"])

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return {
        "status": "added",
        "chunks_added": len(chunks),
        "doc_id_prefix": doc_id_prefix
    }