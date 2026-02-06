from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body
from typing import List, Annotated

from app.schemas.rag import IngestResponse, QueryRequest, QueryResponse
from app.services.ingestion import ingestion_service
from app.services.generation import rag_service

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    file_id: str = Form(...)
):
    """
    Ingest a file (PDF or Text).
    """
    content = await file.read()
    message = ingestion_service.ingest_file(content, file.filename, user_id, file_id)
    return {"message": message}


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system.
    """
    result = await rag_service.query(request.query, request.user_id)
    return result
