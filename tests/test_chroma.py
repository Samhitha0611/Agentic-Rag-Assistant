import chromadb


# Create an in-memory Chroma client
client = chromadb.Client()


# Create a collection
collection = client.get_or_create_collection(
    name="rag_test"
)


# Add documents
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "Machine learning allows computers to learn patterns from data.",
        "Deep learning uses neural networks to solve complex problems.",
        "Chicken biryani is a popular rice dish."
    ]
)


# Search the collection
results = collection.query(
    query_texts=["How do computers learn from data?"],
    n_results=2
)


print("\nQuery:")
print("How do computers learn from data?")


print("\nRetrieved documents:")

for document in results["documents"][0]:
    print("-", document)