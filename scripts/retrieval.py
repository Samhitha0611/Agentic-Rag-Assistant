from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

chroma_dir = project_root / "data" / "chroma_db"

# --------------------------------------------------
# 2. Create the embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# 3. Load the existing ChromaDB
# --------------------------------------------------

vector_store = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory=str(chroma_dir),
)

# --------------------------------------------------
# 4. Retrieval function
# --------------------------------------------------

def retrieve_documents(query, k=5):
    """
    Retrieve relevant document chunks for a given user query.
    """

    results = vector_store.similarity_search_with_score(
        query,
        k=k,
    )

    if not results:
        return []

    # Print scores for debugging
    for document, score in results:
        print("\nRetrieved document:")
        print(document.page_content[:200])
        print("Score:", score)
        print("Source:", document.metadata.get("source_file"))

    return results