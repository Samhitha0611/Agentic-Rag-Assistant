from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Project root directory
project_root = Path(__file__).resolve().parent.parent

# PDF location
pdf_path = project_root / "data" / "documents" / "sample.pdf"


# 1. Load the PDF
loader = PyPDFLoader(str(pdf_path))

documents = loader.load()


print("PDF loaded successfully!")
print("Number of pages:", len(documents))


# 2. Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


# 3. Split the documents into chunks
chunks = text_splitter.split_documents(documents)


print("\nDocument splitting completed!")
print("Number of chunks:", len(chunks))


# 4. Display each chunk
for i, chunk in enumerate(chunks, start=1):

    print(f"\n--- Chunk {i} ---")

    print("Content:")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)