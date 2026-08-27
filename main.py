import os, io, json, urllib.request
import numpy as np
import torch, open_clip
from PIL import Image
from fastapi import FastAPI, UploadFile, File

SERVER = "https://onetap-1opu-production.up.railway.app"
GAME = os.environ.get("GAME", "pokemon")   # which game this service handles
   DATA_DIR = "/data"
   VEC_FILE = f"{DATA_DIR}/{GAME}_vectors.npy"
   META_FILE = f"{DATA_DIR}/{GAME}_meta.json"
   PROGRESS = f"{DATA_DIR}/{GAME}_progress.json"

print("Loading CLIP (CPU)...")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.to("cpu").eval()

def fp(img):
    x = preprocess(img).unsqueeze(0).to("cpu")
    with torch.no_grad():
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()[0]

def game_tag(setlist):
    games = set(str(s.get("game","")).lower() for s in setlist)
    if GAME == "pokemon": return "pokemon"
    if GAME == "onepiece": return next((g for g in games if "piece" in g), "one_piece")
    return GAME

def build_index():
    print(f"Building {GAME} index from server (first boot, this takes a while)...")
    with urllib.request.urlopen(f"{SERVER}/api/sets") as r:
        setlist = json.load(r)
    setlist = setlist.get("sets", setlist) if isinstance(setlist, dict) else setlist
    tag = game_tag(setlist)
    sets = [s for s in setlist if str(s.get("game","")).lower() == tag]
    done = set(json.load(open(PROGRESS))) if os.path.exists(PROGRESS) else set()
    meta = json.load(open(META_FILE, encoding="utf-8")) if os.path.exists(META_FILE) else []
    vecs = list(np.load(VEC_FILE)) if os.path.exists(VEC_FILE) else []
    for s in sets:
        sid = str(s.get("id"))
        if sid in done: continue
        try:
            with urllib.request.urlopen(f"{SERVER}/api/sets/{sid}/cards", timeout=30) as r:
                data = json.load(r)
            cards = data.get("cards", data) if isinstance(data, dict) else data
            cards = [c for c in cards if c.get("image")]
        except Exception:
            continue
        for c in cards:
            try:
                req = urllib.request.Request(c["image"], headers={"User-Agent":"Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    img = Image.open(io.BytesIO(resp.read())).convert("RGB")
                vecs.append(fp(img))
                meta.append({"name": c.get("name"), "number": c.get("number"), "set": s.get("name")})
            except Exception:
                pass
        done.add(sid)
        np.save(VEC_FILE, np.array(vecs))
        json.dump(meta, open(META_FILE,"w",encoding="utf-8"))
        json.dump(list(done), open(PROGRESS,"w"))
        print(f"  {s.get('name')}: total {len(meta)}")
    return np.array(vecs), meta

if os.path.exists(VEC_FILE) and os.path.exists(META_FILE):
    print("Loading saved index...")
    vectors = np.load(VEC_FILE)
    meta = json.load(open(META_FILE, encoding="utf-8"))
else:
    vectors, meta = build_index()
print(f"Ready: {len(meta)} {GAME} cards indexed.")

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok", "game": GAME, "cards": len(meta)}

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    q = fp(img)
    sims = vectors @ q
    order = sims.argsort()[::-1][:5]
    return {"matches": [
        {"name": meta[i]["name"], "number": meta[i].get("number"),
         "set": meta[i].get("set"), "score": float(sims[i])}
        for i in order
    ]}
