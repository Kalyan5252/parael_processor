from pydantic import BaseModel
from typing import List, Optional

class IngestResponse(BaseModel):
    message: str
    doc_ids: Optional[List[str]] = None

class QueryRequest(BaseModel):
    query: str
    user_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
