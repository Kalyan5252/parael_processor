import os
import torch
import clip
import chromadb
from PIL import Image
import io
import uuid
import numpy as np

from app.core.config import settings

class ClipService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.chroma_client = None
        self.collection = None
        self.collection_name = 'clip_test'
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        
        print("Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        print("Connecting to ChromaDB Cloud...")
        self.chroma_client = chromadb.CloudClient(
            api_key=settings.CHROMA_KEY, # Using config
            tenant=settings.CHROMA_TENANT,
            database=settings.CHROMA_DATABASE
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._initialized = True

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(image_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def get_text_embedding(self, text: str) -> np.ndarray:
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def index_image(self, file_content: bytes, filename: str, user_id: str, file_id: str) -> str:
        self.initialize()
        
        image = Image.open(io.BytesIO(file_content)).convert("RGB")
        emb = self.get_image_embedding(image)
        
        doc_id = str(uuid.uuid4())
        
        self.collection.add(
            embeddings=[emb.tolist()],
            ids=[doc_id],
            metadatas=[{"path": filename, "userId": user_id, "fileId": file_id}]
        )
        return doc_id

    def search(self, query: str, user_id: str, top_k: int = 5) -> dict:
        self.initialize()
        
        query_emb = self.get_text_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            where={"userId": user_id}
        )
        
        search_results = []
        file_ids = []
        if results["metadatas"] and len(results["metadatas"]) > 0:
            for item, item_id in zip(results["metadatas"][0], results["ids"][0]):
                search_results.append({
                    "id": item_id,
                    "path": item.get("path", "")
                })
                if item.get("fileId"):
                    file_ids.append(item.get("fileId"))
        return {
            "results": search_results,
            "file_ids": file_ids
        }

clip_service = ClipService()