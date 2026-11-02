from langchain_core.documents import Document

from scripts.context_builder import build_context


def test_build_context_single_document():
    document = Document(
        page_content="Python is a programming language.",
        metadata={
            "source_file": "sample.pdf",
            "page_label": "Page 1",
        },
    )

    results = [(document, 0.95)]

    context = build_context(results)

    assert "--- Retrieved Chunk 1 ---" in context
    assert "Source: sample.pdf" in context
    assert "Page: Page 1" in context
    assert "Python is a programming language." in context


def test_build_context_multiple_documents():
    document1 = Document(
        page_content="Python is used for AI and machine learning.",
        metadata={
            "source_file": "sample.pdf",
            "page_label": "Page 1",
        },
    )

    document2 = Document(
        page_content="RAG combines retrieval with generation.",
        metadata={
            "source_file": "sample.pdf",
            "page_label": "Page 2",
        },
    )

    results = [
        (document1, 0.95),
        (document2, 0.87),
    ]

    context = build_context(results)

    assert "--- Retrieved Chunk 1 ---" in context
    assert "--- Retrieved Chunk 2 ---" in context
    assert "Python is used for AI and machine learning." in context
    assert "RAG combines retrieval with generation." in context


def test_build_context_missing_metadata():
    document = Document(
        page_content="This document has missing metadata.",
        metadata={},
    )

    results = [(document, 0.80)]

    context = build_context(results)

    assert "Source: Unknown source" in context
    assert "Page: Unknown page" in context
    assert "This document has missing metadata." in context
