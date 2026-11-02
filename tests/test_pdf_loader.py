from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


# Project root directory
project_root = Path(__file__).resolve().parent.parent

# PDF location
pdf_path = project_root / "data" / "documents" / "sample.pdf"


# Create the PDF loader
loader = PyPDFLoader(str(pdf_path))


# Load the PDF
documents = loader.load()


# Display basic information
print("PDF loaded successfully!")
print("Number of pages:", len(documents))


# Display information about the first page
first_page = documents[0]

print("\n--- First Page ---")
print("Content:")
print(first_page.page_content[:1000])

print("\nMetadata:")
print(first_page.metadata)