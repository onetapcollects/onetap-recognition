import os, io, json, urllib.request, requests
import numpy as np
import torch, open_clip
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form

# All five games in one service. CLIP loads once; each game's index loads into memory.
# A scan says which game it is, and we search ONLY that game's index (no cross-game misreads).

DATA_DIR = "/data"
RELEASE = "https://github.com/onetapcollects/onetap-recognition/releases/download/indexes"

# App sends these cardType names. Map each to its index file stem.
# NOTE: app sends "one_piece" (underscore) but the files are "onepiece" (no underscore).
GAME_FILES = {
    "pokemon":   "pokemon",
    "magic":     "magic",
    "yugioh":    "yugioh",
    "one_piece": "onepiece",
    "lorcana":   "lorcana",
}

os.makedirs(DATA_DIR, exist_ok=True)

def ensure_file(stem, ext):
    """Download <stem>_<ext> from the GitHub Release into /data if not already there."""
    fname = f"{stem}_{ext}"
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"{RELEASE}/{fname}"
    print(f"Downloading {fname} ...")
    with requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True, timeout=300, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    print(f"  saved {fname} ({os.path.getsize(path)} bytes)")
    return path

print("Loading CLIP (CPU)...")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.to("cpu").eval()

# Load every game: download its two files, then read them into memory.
INDEX = {}  # cardType -> {"vectors": np.array, "meta": list}
for cardType, stem in GAME_FILES.items():
    try:
        vec_path = ensure_file(stem, "vectors.npy")
        meta_path = ensure_file(stem, "meta.json")
        vectors = np.load(vec_path)
        meta = json.load(open(meta_path, encoding="utf-8"))
        INDEX[cardType] = {"vectors": vectors, "meta": meta}
        print(f"Loaded {cardType}: {len(meta)} cards.")
    except Exception as e:
        print(f"FAILED to load {cardType}: {e}")

total = sum(len(v["meta"]) for v in INDEX.values())
print(f"Ready: {len(INDEX)} games, {total} cards total.")

def fp(img):
    x = preprocess(img).unsqueeze(0).to("cpu")
    with torch.no_grad():
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()[0]

app = FastAPI()

@app.get("/")
def health():
    return {
        "status": "ok",
        "games": {g: len(v["meta"]) for g, v in INDEX.items()},
        "total": total,
    }

@app.post("/identify")
async def identify(file: UploadFile = File(...), game: str = Form(...)):
    if game not in INDEX:
        return {"error": f"unknown game '{game}'", "known": list(INDEX.keys())}
    data = INDEX[game]
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    q = fp(img)
    sims = data["vectors"] @ q
    order = sims.argsort()[::-1][:5]
    m = data["meta"]
    return {"game": game, "matches": [
        {"name": m[i]["name"], "number": m[i].get("number"),
         "set": m[i].get("set"), "score": float(sims[i])}
        for i in order
    ]}
