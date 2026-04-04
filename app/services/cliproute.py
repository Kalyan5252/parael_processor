import os
import torch
import clip
import chromadb
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

IMAGE_FOLDER = "./images"  # put your images here
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = clip.load("ViT-B/32", device=DEVICE)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="clip_images")


def get_image_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(image)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def get_text_embedding(text):
    tokens = clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]



def index_images():
    print("Indexing images...")

    for filename in os.listdir(IMAGE_FOLDER):
        path = os.path.join(IMAGE_FOLDER, filename)

        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        try:
            emb = get_image_embedding(path)

            collection.add(
                embeddings=[emb],
                ids=[filename],
                metadatas=[{"path": path}]
            )

            print(f"Indexed: {filename}")

        except Exception as e:
            print(f"Error with {filename}: {e}")



app = FastAPI()


class Query(BaseModel):
    query: str


@app.on_event("startup")
def startup_event():
    index_images()


@app.post("/search")
def search(q: Query):
    query_emb = get_text_embedding(q.query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=5
    )

    return {
        "query": q.query,
        "results": results["metadatas"][0]
    }