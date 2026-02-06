import google.generativeai as genai
import chromadb
from app.core.config import settings

# Configure Google Gemini
genai.configure(api_key=settings.GOOGLE_API_KEY)

# Configure Chroma
chroma_client = chromadb.CloudClient(
    api_key=settings.CHROMA_KEY,
    tenant=settings.CHROMA_TENANT,
    database=settings.CHROMA_DATABASE
)

collection = chroma_client.get_or_create_collection(
    name=settings.CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

EMBEDDING_MODEL = "models/gemini-embedding-001"

GENERATION_MODEL = "models/gemini-flash-lite-latest"

def embed_text(texts: list[str]) -> list[list[float]]:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="retrieval_document",
    )
    return result["embedding"]
