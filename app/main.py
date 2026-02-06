from fastapi import FastAPI
from app.core.config import settings
from app.api import endpoints

app = FastAPI(title=settings.PROJECT_NAME)
# app = FastAPI(title='test')

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(endpoints.router, prefix=settings.API_V1_STR)

@app.get("/health")
def health_check():
    return {"status": "ok"}
