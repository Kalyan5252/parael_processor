from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body
from typing import List, Annotated

from app.schemas.rag import IngestResponse, QueryRequest, QueryResponse
from app.schemas.clip import ImageSearchRequest, ImageSearchResponse, ImageIndexResponse
from app.services.ingestion import ingestion_service
from app.services.generation import rag_service
from app.services.clipservice import clip_service

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


@router.post("/clip/index", response_model=ImageIndexResponse)
async def clip_index_image(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    file_id: str = Form(...)
):
    """
    Index an image using CLIP and ChromaDB.
    """
    content = await file.read()
    try:
        doc_id = clip_service.index_image(content, file.filename, user_id, file_id)
        return {"message": "Image indexed successfully", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clip/search", response_model=ImageSearchResponse)
async def clip_search(request: ImageSearchRequest):
    """
    Search for images using text query via CLIP.
    """
    try:
        search_output = clip_service.search(request.query, request.user_id, request.top_k)
        return {"results": search_output["results"], "file_ids": search_output["file_ids"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
