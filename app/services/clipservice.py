import os
import torch
import clip
import chromadb
from PIL import Image

IMAGE_FOLDER = "./images"   # put test images here
QUERY = "a white car"       # 🔥 change this to test
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP...")
model, preprocess = clip.load("ViT-B/32", device=device)

chroma_client = chromadb.CloudClient(
    api_key='ck-5ihVuj8ehAML8zZQKNWA49XQbTvU47i33SL2jguiHKtH',
    tenant="851123fe-e1f1-46f9-8225-4d0543e2d988",
    database="test"
)

collection = chroma_client.get_or_create_collection(
    name='clip_test',
    metadata={"hnsw:space": "cosine"}
)

def get_image_embedding(path):
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def get_text_embedding(text):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]



# print("\nIndexing images...")

# image_paths = []
# embeddings = []
# ids = []

# for i, filename in enumerate(os.listdir('/Users/kalyan/Desktop/MY FILES/DT projects/smartcloudpip/app/services/images/')):
#     if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
#         continue

#     path = os.path.join('/Users/kalyan/Desktop/MY FILES/DT projects/smartcloudpip/app/services/images/', filename)

#     try:
#         emb = get_image_embedding(path)

#         image_paths.append({"path": path})
#         embeddings.append(emb)
#         ids.append(str(i))

#         print(f"✔ Indexed: {filename}")

#     except Exception as e:
#         print(f"❌ Error: {filename} -> {e}")

# # Add to Chroma
# collection.add(
#     embeddings=embeddings,
#     ids=ids,
#     metadatas=image_paths
# )


print(f"\n🔍 Searching for: '{QUERY}'")

query_emb = get_text_embedding(QUERY)

results = collection.query(
    query_embeddings=[query_emb],
    n_results=TOP_K
)

print("\n🎯 Top Results:")

for i, item in enumerate(results["metadatas"][0]):
    print(f"{i+1}. {item['path']}")