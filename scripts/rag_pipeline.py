from .retrieval import retrieve_documents
from .context_builder import build_context
from .llm import client, MODEL_NAME

# ---------------------------------------------------
# 1. Create RAG prompt
# ---------------------------------------------------

def create_rag_prompt(query, context):
    """
    Create a prompt containing the user query
    and the retrieved document context.
    """

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided documents."

Do not make up information.

Context:
{context}

User Question:
{query}

Answer:
"""

    return prompt

# ---------------------------------------------------
# 2. Generate RAG response
# ---------------------------------------------------

def generate_rag_response(query, k=5):
    """
    Generate an answer using the RAG pipeline.

    Steps:
    1. Retrieve relevant documents.
    2. Build context from retrieved documents.
    3. Create the RAG prompt.
    4. Send the prompt to the LLM.
    5. Return the generated answer.
    """

    try:
        # Step 1: Retrieve relevant documents
        results = retrieve_documents(query, k=k)

        # Step 2: Check if relevant documents were found
        if not results:
            return (
                "I could not find relevant information in the provided documents.",
                []
            )

        # Step 3: Build context
        context = build_context(results)

        # Step 4: Check if context was created successfully
        if not context or not context.strip():
            return (
                "I could not build enough context from the retrieved documents.",
                []
            )

        # Step 5: Create prompt
        prompt = create_rag_prompt(query, context)

        # Step 6: Generate response using Gemini
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        # Step 7: Check if Gemini returned a valid response
        if not response or not response.text:
            return (
                "I was unable to generate a response. Please try again.",
                []
            )

        return response.text, results

    except Exception as e:
        print("\n========== RAG ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("================================\n")

        return (
            f"An error occurred while processing your question: {str(e)}",
            []
        )