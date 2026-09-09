import os, io, json, urllib.request, requests
import numpy as np
import torch, open_clip
import imagehash
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form

# All five games in one service. CLIP loads once; each game's index loads into memory.
# NOW ALSO: perceptual hashes as a SECOND signal. CLIP is great but weak on some cards (busy
# foils, promos, reprints, and foreign cards). Hashing catches those - it matches on artwork,
# so a Japanese/Chinese/Korean card matches its English equivalent. The two signals are fused
# so where one is weak the other backs it up. (Sept 2026)

DATA_DIR = "/data"
RELEASE = "https://github.com/onetapcollects/onetap-recognition/releases/download/indexes"

GAME_FILES = {
    "pokemon":   "pokemon",
    "magic":     "magic",
    "yugioh":    "yugioh",
    "one_piece": "onepiece",
    "lorcana":   "lorcana",
}

os.makedirs(DATA_DIR, exist_ok=True)

def ensure_file(fname):
    """Download fname from the GitHub Release into /data if not already there."""
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"{RELEASE}/{fname}"
    print(f"Downloading {fname} ...")
    with requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True, timeout=600, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    print(f"  saved {fname} ({os.path.getsize(path)} bytes)")
    return path

print("Loading CLIP (CPU)...")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.to("cpu").eval()

INDEX = {}  # cardType -> {"vectors", "meta", "hashes"(optional)}
for cardType, stem in GAME_FILES.items():
    entry = {}
    try:
        vec_path = ensure_file(f"{stem}_vectors.npy")
        meta_path = ensure_file(f"{stem}_meta.json")
        entry["vectors"] = np.load(vec_path)
        entry["meta"] = json.load(open(meta_path, encoding="utf-8"))
        print(f"Loaded {cardType} fingerprints: {len(entry['meta'])} cards.")
    except Exception as e:
        print(f"FAILED to load {cardType} fingerprints: {e}")
        continue
    # hashes are optional - load if present, precompute hash objects for speed
    try:
        hash_path = ensure_file(f"{stem}_hashes.json")
        raw = json.load(open(hash_path, encoding="utf-8"))
        hlist = []
        for h in raw:
            try:
                hlist.append({
                    "name": h.get("name"), "number": h.get("number"), "set": h.get("set"),
                    "p": imagehash.hex_to_hash(h["phash"]),
                    "a": imagehash.hex_to_hash(h["ahash"]),
                    "d": imagehash.hex_to_hash(h["dhash"]),
                    "w": imagehash.hex_to_hash(h["whash"]),
                })
            except Exception:
                pass
        entry["hashes"] = hlist
        print(f"Loaded {cardType} hashes: {len(hlist)} cards.")
    except Exception as e:
        print(f"No hashes for {cardType} (that's ok): {e}")
    INDEX[cardType] = entry

total = sum(len(v["meta"]) for v in INDEX.values())
print(f"Ready: {len(INDEX)} games, {total} cards total.")

def fp(img):
    x = preprocess(img).unsqueeze(0).to("cpu")
    with torch.no_grad():
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()[0]

def hash_match(img, hashes, topn=5):
    """Return the best hash matches (lower distance = better)."""
    t_p = imagehash.phash(img); t_a = imagehash.average_hash(img)
    t_d = imagehash.dhash(img); t_w = imagehash.whash(img)
    scored = []
    for e in hashes:
        dist = (t_p - e["p"]) + (t_a - e["a"]) + (t_d - e["d"]) + (t_w - e["w"])
        scored.append((dist, e))
    scored.sort(key=lambda x: x[0])
    return scored[:topn]

app = FastAPI()

@app.get("/")
def health():
    return {
        "status": "ok",
        "games": {g: {"cards": len(v["meta"]), "hashes": len(v.get("hashes", []))} for g, v in INDEX.items()},
        "total": total,
    }

@app.post("/identify")
async def identify(file: UploadFile = File(...), game: str = Form(...)):
    if game not in INDEX:
        return {"error": f"unknown game '{game}'", "known": list(INDEX.keys())}
    data = INDEX[game]
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # --- Signal 1: CLIP fingerprint ---
    q = fp(img)
    sims = data["vectors"] @ q
    clip_order = sims.argsort()[::-1][:5]
    m = data["meta"]
    clip_top_score = float(sims[clip_order[0]]) if len(clip_order) else 0.0
    clip_matches = [
        {"name": m[i]["name"], "number": m[i].get("number"), "set": m[i].get("set"),
         "score": float(sims[i]), "source": "clip"}
        for i in clip_order
    ]

    # --- Signal 2: perceptual hash (if available) ---
    hash_matches = []
    hashes = data.get("hashes")
    if hashes:
        for dist, e in hash_match(img, hashes, topn=5):
            # convert distance to a 0-1 confidence: dist 0 = 1.0, dist ~120 = 0
            conf = max(0.0, 1.0 - (dist / 120.0))
            hash_matches.append({
                "name": e["name"], "number": e["number"], "set": e["set"],
                "distance": int(dist), "score": conf, "source": "hash",
            })

    # --- Fuse: if CLIP is confident, trust it; if CLIP is weak but hash is strong, prefer hash ---
    # CLIP score is a cosine sim (higher better, ~0.8+ = confident). Hash conf is 0-1 (from distance).
    best = None
    if clip_matches:
        best = clip_matches[0]
    if hash_matches:
        h0 = hash_matches[0]
        # if CLIP is weak (<0.75) and the hash match is strong (distance small / conf high), use hash
        if clip_top_score < 0.75 and h0["score"] > 0.80:
            best = h0

    return {
        "game": game,
        "best": best,
        "clip": clip_matches,
        "hash": hash_matches,
    }
