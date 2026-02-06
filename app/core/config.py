from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI RAG App"
    API_V1_STR: str = "/api/v1"
    
    # Google Gemini
    GOOGLE_API_KEY: str

    # Chroma Cloud
    CHROMA_KEY: str
    CHROMA_TENANT: str = "851123fe-e1f1-46f9-8225-4d0543e2d988"
    CHROMA_DATABASE: str = "test"
    CHROMA_COLLECTION: str = "rag_context_embeddings"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
