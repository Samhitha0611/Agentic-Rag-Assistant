from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
Artificial Intelligence is a field of computer science.

Machine Learning is a subset of Artificial Intelligence.
It allows computers to learn patterns from data.

Deep Learning is a subset of Machine Learning.
It uses neural networks to learn complex patterns.

Natural Language Processing helps computers understand
and process human language.
"""


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_text(text)


for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)