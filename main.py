import os, io, json, urllib.request, requests
import numpy as np
import torch, open_clip
import imagehash
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form

# One recognition service for everything - to avoid paying for a second Railway service.
#  - English cards: CLIP fingerprints (+ hash second signal), as before.
#  - FOREIGN cards (Japanese/Chinese/Korean): CollectorVision artwork embedding matched against an
#    English reference catalogue, NARROWED by the OCR-read collector number. Foreign-only endpoint.
# Memory-conscious: the foreign catalogue is stored/loaded as float16 to halve its RAM footprint.

DATA_DIR = "/data"
RELEASE = "https://github.com/onetapcollects/onetap-recognition/releases/download/indexes"
GAME_FILES = {"pokemon":"pokemon","magic":"magic","yugioh":"yugioh","one_piece":"onepiece","lorcana":"lorcana"}
os.makedirs(DATA_DIR, exist_ok=True)

def ensure_file(fname):
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"{RELEASE}/{fname}"
    print(f"Downloading {fname} ...")
    with requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, allow_redirects=True, timeout=600, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
    print(f"  saved {fname} ({os.path.getsize(path)} bytes)")
    return path

print("Loading CLIP (CPU)...")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.to("cpu").eval()

INDEX = {}
for cardType, stem in GAME_FILES.items():
    entry = {}
    try:
        entry["vectors"] = np.load(ensure_file(f"{stem}_vectors.npy"))
        entry["meta"] = json.load(open(ensure_file(f"{stem}_meta.json"), encoding="utf-8"))
        print(f"Loaded {cardType} fingerprints: {len(entry['meta'])} cards.")
    except Exception as e:
        print(f"FAILED {cardType} fingerprints: {e}"); continue
    try:
        raw = json.load(open(ensure_file(f"{stem}_hashes.json"), encoding="utf-8"))
        hlist = []
        for h in raw:
            try:
                hlist.append({"name":h.get("name"),"number":h.get("number"),"set":h.get("set"),
                    "p":imagehash.hex_to_hash(h["phash"]),"a":imagehash.hex_to_hash(h["ahash"]),
                    "d":imagehash.hex_to_hash(h["dhash"]),"w":imagehash.hex_to_hash(h["whash"])})
            except Exception: pass
        entry["hashes"] = hlist
        print(f"Loaded {cardType} hashes: {len(hlist)} cards.")
    except Exception as e:
        print(f"No hashes for {cardType}: {e}")
    INDEX[cardType] = entry

total = sum(len(v["meta"]) for v in INDEX.values())
print(f"Ready: {len(INDEX)} games, {total} cards.")

# ---- FOREIGN catalogue (CollectorVision), loaded lazily to save memory until first foreign scan ----
FOREIGN = {"loaded": False, "embs": None, "meta": None, "norms": None, "num_index": None, "embedder": None}
def _digits(s):
    return "".join(ch for ch in str(s or "").split("/")[0] if ch.isdigit())
def load_foreign():
    if FOREIGN["loaded"]: return
    try:
        import collector_vision as cvg
        cat = cvg.CatalogV2.load("pokemon", include_metadata=True)
        FOREIGN["embedder"] = cat.embedder
        d = np.load(ensure_file("foreign_ref.npz"), allow_pickle=True)
        embs = d["embeddings"].astype(np.float16)  # float16 halves memory
        meta = json.load(open(ensure_file("foreign_ref_meta.json"), encoding="utf-8"))
        norms = np.linalg.norm(embs.astype(np.float32), axis=1) + 1e-9
        num_index = {}
        for i, m in enumerate(meta):
            num_index.setdefault(_digits(m.get("number")), []).append(i)
        FOREIGN.update({"loaded":True,"embs":embs,"meta":meta,"norms":norms,"num_index":num_index})
        print(f"Foreign catalogue loaded: {len(meta)} refs.")
    except Exception as e:
        print(f"Foreign load failed: {e}")

def fp(img):
    x = preprocess(img).unsqueeze(0).to("cpu")
    with torch.no_grad():
        v = model.encode_image(x); v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()[0]

def preprocess_for_hash(img):
    try:
        gray = img.convert("L"); bw = ImageOps.autocontrast(gray)
        bbox = bw.point(lambda p: 255 if p < 245 else 0).getbbox()
        if bbox:
            l,t,r,b = bbox; w,h = r-l,b-t
            if w>20 and h>20:
                px,py=int(w*.02),int(h*.02); img=img.crop((l+px,t+py,r-px,b-py))
    except Exception: pass
    return img.convert("RGB").resize((256,256))

def hash_match(img, hashes, topn=5):
    pimg = preprocess_for_hash(img)
    tp,ta,td,tw = imagehash.phash(pimg),imagehash.average_hash(pimg),imagehash.dhash(pimg),imagehash.whash(pimg)
    scored = [((tp-e["p"])+(ta-e["a"])+(td-e["d"])+(tw-e["w"]), e) for e in hashes]
    scored.sort(key=lambda x:x[0]); return scored[:topn]

app = FastAPI()

@app.get("/")
def health():
    return {"status":"ok","games":{g:{"cards":len(v["meta"]),"hashes":len(v.get("hashes",[]))} for g,v in INDEX.items()},"total":total,"foreign_loaded":FOREIGN["loaded"]}

@app.post("/identify")
async def identify(file: UploadFile = File(...), game: str = Form(...)):
    if game not in INDEX:
        return {"error":f"unknown game '{game}'","known":list(INDEX.keys())}
    data = INDEX[game]
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    q = fp(img); sims = data["vectors"] @ q
    order = sims.argsort()[::-1][:5]; m = data["meta"]
    clip_top = float(sims[order[0]]) if len(order) else 0.0
    clip = [{"name":m[i]["name"],"number":m[i].get("number"),"set":m[i].get("set"),"score":float(sims[i]),"source":"clip"} for i in order]
    hm = []
    if data.get("hashes"):
        for dist,e in hash_match(img, data["hashes"], 5):
            hm.append({"name":e["name"],"number":e["number"],"set":e["set"],"distance":int(dist),"score":max(0.0,1.0-dist/120.0),"source":"hash"})
    best = clip[0] if clip else None
    if hm and clip_top < 0.75 and hm[0]["score"] > 0.80:
        best = hm[0]
    return {"game":game,"best":best,"clip":clip,"hash":hm}

@app.post("/identify_foreign")
async def identify_foreign(file: UploadFile = File(...), number: str = Form(default="")):
    load_foreign()
    if not FOREIGN["loaded"]:
        return {"best":None,"confident":False,"error":"foreign catalogue unavailable"}
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    q = np.asarray(FOREIGN["embedder"].embed(img), dtype=np.float32); qn = np.linalg.norm(q)+1e-9
    tgt = _digits(number)
    idxs = FOREIGN["num_index"].get(tgt) if tgt else None
    narrowed = bool(idxs)
    if not idxs: idxs = list(range(len(FOREIGN["meta"])))
    embs = FOREIGN["embs"]; norms = FOREIGN["norms"]; meta = FOREIGN["meta"]
    sims = []
    for i in idxs:
        e = embs[i].astype(np.float32)
        sims.append((float(np.dot(e,q)/(norms[i]*qn)), i))
    sims.sort(reverse=True)
    top = [{"name":meta[i].get("name"),"set":meta[i].get("set"),"number":meta[i].get("number"),"score":round(s,4)} for s,i in sims[:5]]
    best = top[0] if top else None
    confident = False
    if best and narrowed:
        gap = (top[0]["score"]-top[1]["score"]) if len(top)>1 else 0.2
        confident = best["score"]>=0.32 and gap>=0.03
    return {"best":best if confident else None,"confident":confident,"narrowed":narrowed,"matches":top}
