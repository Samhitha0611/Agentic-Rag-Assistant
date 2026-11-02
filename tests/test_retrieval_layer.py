from retrieval import retrieve_documents


# --------------------------------------------------
# 1. Test query
# --------------------------------------------------

query = "What is the capital of France?"


# --------------------------------------------------
# 2. Retrieve documents
# --------------------------------------------------

results = retrieve_documents(
    query,
    k=2,
)


# --------------------------------------------------
# 3. Display results
# --------------------------------------------------

print("\n========================================")
print("RETRIEVAL LAYER TEST")
print("========================================")

print("\nQuery:")
print(query)

print("\nRetrieved documents:")

for i, (document, score) in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")

    print("Content:")
    print(document.page_content)

    print("\nSimilarity Score:")
    print(score)

    print("\nMetadata:")
    print(document.metadata)


print("\n========================================")
print("RETRIEVAL LAYER TEST COMPLETED")
print("========================================")