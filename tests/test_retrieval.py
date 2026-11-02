from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

chroma_dir = project_root / "data" / "chroma_db"

# --------------------------------------------------
# 2. Create the same embedding model
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
# 4. User query
# --------------------------------------------------

query = "What is the best recipe for biryani?"

# --------------------------------------------------
# 5. Perform similarity search
# --------------------------------------------------
results = vector_store.similarity_search_with_score(
    query,
    k=2,
)
# --------------------------------------------------
# 6. Display results
# --------------------------------------------------
print("\n========================================")
print("RETRIEVAL TEST")
print("========================================")

print("\nQuery:")
print(query)

print("\nRetrieved documents:")

for i, (document, score) in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")

    print("Content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)

    print("\nSimilarity Score:")
    print(score)

print("\n========================================")
print("RETRIEVAL COMPLETED")
print("========================================")