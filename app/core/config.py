from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI RAG App"
    API_V1_STR: str = "/api/v1"
    
    # Google Gemini
    GOOGLE_API_KEY: str

    # Chroma Cloud
    CHROMA_KEY: str
    CHROMA_TENANT: str
    CHROMA_DATABASE: str
    CHROMA_COLLECTION: str

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
