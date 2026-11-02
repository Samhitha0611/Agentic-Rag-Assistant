from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# 1. Create our embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Create/load a persistent Chroma database
vector_store = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="./data/chroma_db",
)


# 3. Create sample documents
documents = [
    Document(
        page_content="Machine learning allows computers to learn patterns from data.",
        metadata={"topic": "machine learning"}
    ),
    Document(
        page_content="Deep learning uses neural networks to learn complex patterns.",
        metadata={"topic": "deep learning"}
    ),
    Document(
        page_content="Natural Language Processing helps computers understand human language.",
        metadata={"topic": "NLP"}
    ),
    Document(
        page_content="Chicken biryani is a popular rice dish.",
        metadata={"topic": "food"}
    ),
]


# 4. Add documents to Chroma
vector_store.add_documents(documents)


# 5. Search for relevant documents
query = "How do computers learn from data?"

results = vector_store.similarity_search(
    query,
    k=2
)


print("\nQuery:")
print(query)

print("\nRetrieved documents:")

for i, document in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Content:", document.page_content)
    print("Metadata:", document.metadata)