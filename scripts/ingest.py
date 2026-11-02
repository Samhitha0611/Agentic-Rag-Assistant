from pathlib import Path
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest_documents(clear_existing=True):

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

    project_root = Path(__file__).resolve().parent.parent

    documents_dir = project_root / "data" / "documents"
    chroma_dir = project_root / "data" / "chroma_db"

# --------------------------------------------------
# 2. Create embedding model
# --------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# --------------------------------------------------
# 3. Create/load persistent ChromaDB
# --------------------------------------------------

    vector_store = Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )

    def clear_vector_store():
        """
        Remove all existing documents from ChromaDB.
        """

        existing_data = vector_store.get()

        existing_ids = existing_data.get("ids", [])

        if existing_ids:
            vector_store.delete(ids=existing_ids)
            print(f"Cleared {len(existing_ids)} old chunks from ChromaDB.")
        else:
            print("ChromaDB is already empty.")

    # Clear old document data when processing a new upload
    if clear_existing:
        existing_data = vector_store.get()
        existing_ids = existing_data.get("ids", [])

        if existing_ids:
            vector_store.delete(ids=existing_ids)
            print(
                f"Cleared {len(existing_ids)} old chunks from ChromaDB."
            )
        else:
            print("ChromaDB is already empty.")

# --------------------------------------------------
# 4. Text splitter
# --------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

# --------------------------------------------------
# 5. Find PDF documents
# --------------------------------------------------

    pdf_files = list(documents_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in:", documents_dir)
        exit()

    print(f"Found {len(pdf_files)} PDF file(s).")

# --------------------------------------------------
# 6. Process each PDF
# --------------------------------------------------

    all_chunks = []

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        # Load PDF
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        print(f"Pages loaded: {len(documents)}")

        # Split into chunks
        chunks = text_splitter.split_documents(documents)

        print(f"Chunks created: {len(chunks)}")

        # Add custom metadata
        for chunk in chunks:
            chunk.metadata["source_file"] = pdf_path.name

        all_chunks.extend(chunks)

# --------------------------------------------------
# 7. Create unique IDs for chunks
# --------------------------------------------------

    existing_data = vector_store.get()

    existing_ids = set(existing_data["ids"])

    new_chunks = []
    new_ids = []

    for chunk in all_chunks:

        source = chunk.metadata.get("source_file", "")
        page = chunk.metadata.get("page", "")
        content = chunk.page_content

        unique_string = f"{source}-{page}-{content}"

        chunk_id = hashlib.md5(
            unique_string.encode("utf-8")
        ).hexdigest()

        if chunk_id not in existing_ids:

            new_chunks.append(chunk)
            new_ids.append(chunk_id)

# --------------------------------------------------
# 8. Store new chunks in ChromaDB
# --------------------------------------------------

    if new_chunks:

        vector_store.add_documents(
            documents=new_chunks,
            ids=new_ids,
        )

        print(f"\nAdded {len(new_chunks)} new chunks to ChromaDB.")

    else:

        print("\nNo new chunks to add.")

# --------------------------------------------------
# 9. Final summary
# --------------------------------------------------

    total_chunks = vector_store._collection.count()

    print("\n----------------------------------------")
    print("INGESTION COMPLETED")
    print("----------------------------------------")
    print(f"PDF files processed : {len(pdf_files)}")
    print(f"New chunks added    : {len(new_chunks)}")
    print(f"Total chunks in DB  : {total_chunks}")
    print("----------------------------------------")

if __name__ == "__main__":
    ingest_documents()