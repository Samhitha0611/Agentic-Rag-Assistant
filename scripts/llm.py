import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ---------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------
# 2. Get Gemini API key
# ---------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set in the environment."
    )


# ---------------------------------------------------
# 3. Create Gemini client
# ---------------------------------------------------

client = genai.Client(api_key=api_key)


# ---------------------------------------------------
# 4. Select the LLM model
# ---------------------------------------------------

MODEL_NAME = "gemini-3.6-flash"

# ---------------------------------------------------
# 5. Test Gemini connection
# ---------------------------------------------------

def test_llm_connection():
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents="Hello! Please reply with: Gemini connection successful."
        )

        print("\nGemini Response:")
        print(response.text)

        print("\nLLM connection successful!")

    except Exception as e:
        print("\nError connecting to Gemini:")
        print(e)


# Run the test
if __name__ == "__main__":
    test_llm_connection()