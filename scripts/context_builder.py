# --------------------------------------------------
# Context Builder
# --------------------------------------------------
def build_context(results):
    """
    Build a context string from retrieved documents.

    Args:
        results: List of (Document, score) tuples.

    Returns:
        A formatted context string.
    """

    context_parts = []

    for i, (document, score) in enumerate(results, start=1):

        source = document.metadata.get(
            "source_file",
            "Unknown source"
        )

        page = document.metadata.get(
            "page_label",
            "Unknown page"
        )

        context_parts.append(
            f"--- Retrieved Chunk {i} ---\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{document.page_content}\n"
        )
        
    return "\n".join(context_parts)