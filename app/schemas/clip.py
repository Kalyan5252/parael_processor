from pydantic import BaseModel
from typing import List, Optional

class ImageSearchRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = 5

class ImageSearchResult(BaseModel):
    path: str
    id: str

class ImageSearchResponse(BaseModel):
    results: List[ImageSearchResult]
    file_ids: List[str]

class ImageIndexResponse(BaseModel):
    message: str
    id: str
