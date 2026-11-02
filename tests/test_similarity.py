from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


sentence_1 = "Machine learning allows computers to learn from data."
sentence_2 = "Artificial intelligence enables computers to learn patterns."
sentence_3 = "I made chicken biryani for dinner."


vector_1 = embeddings.embed_query(sentence_1)
vector_2 = embeddings.embed_query(sentence_2)
vector_3 = embeddings.embed_query(sentence_3)


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    return dot_product / (magnitude_a * magnitude_b)


similarity_1_2 = cosine_similarity(vector_1, vector_2)
similarity_1_3 = cosine_similarity(vector_1, vector_3)


print("Sentence 1:")
print(sentence_1)

print("\nSentence 2:")
print(sentence_2)

print("\nSentence 3:")
print(sentence_3)

print("\nSimilarity between Sentence 1 and Sentence 2:")
print(similarity_1_2)

print("\nSimilarity between Sentence 1 and Sentence 3:")
print(similarity_1_3)