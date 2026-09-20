import os, io, json, requests
import numpy as np
import torch, open_clip
import imagehash
import onnxruntime as ort
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form

# One recognition service for everything (avoids a second Railway service):
#  - English cards: CLIP fingerprints (+ hash second signal).
#  - FOREIGN cards: a lightweight ONNX artwork embedder (the CollectorVision model, run directly via
#    onnxruntime - no heavy package) matched against an English reference catalogue, NARROWED by the
#    OCR-read collector number. Foreign-only. Catalogue loads lazily + as float16 to save memory.

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
        print(f"FAILED {cardType}: {e}"); continue
    try:
        raw = json.load(open(ensure_file(f"{stem}_hashes.json"), encoding="utf-8"))
        hlist=[]
        for h in raw:
            try:
                hlist.append({"name":h.get("name"),"number":h.get("number"),"set":h.get("set"),
                    "p":imagehash.hex_to_hash(h["phash"]),"a":imagehash.hex_to_hash(h["ahash"]),
                    "d":imagehash.hex_to_hash(h["dhash"]),"w":imagehash.hex_to_hash(h["whash"])})
            except Exception: pass
        entry["hashes"]=hlist
        print(f"Loaded {cardType} hashes: {len(hlist)} cards.")
    except Exception as e:
        print(f"No hashes for {cardType}: {e}")
    INDEX[cardType]=entry

total = sum(len(v["meta"]) for v in INDEX.values())
print(f"Ready: {len(INDEX)} games, {total} cards.")

# ---- FOREIGN artwork embedder (CollectorVision's ONNX model, run directly) ----
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FOREIGN = {"loaded":False,"embs":None,"meta":None,"norms":None,"num_index":None,"sess":None,"iname":None,"isize":448}
def _digits(s):
    return "".join(ch for ch in str(s or "").split("/")[0] if ch.isdigit())
def _prep(img, size):
    rgb = img.convert("RGB").resize((size,size), Image.BILINEAR)
    x = np.array(rgb, dtype=np.float32)/255.0
    x = (x - _IMAGENET_MEAN)/_IMAGENET_STD
    return x.transpose(2,0,1)[np.newaxis].astype(np.float32)
def _embed_foreign(img):
    x = _prep(img, FOREIGN["isize"])
    out = FOREIGN["sess"].run(None, {FOREIGN["iname"]: x})[0]
    emb = out.squeeze().astype(np.float32)
    n = float(np.linalg.norm(emb))
    return emb/n if n>1e-8 else emb
def load_foreign():
    if FOREIGN["loaded"]: return
    try:
        model_path = ensure_file("cv_model.onnx")
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        FOREIGN["sess"]=sess
        FOREIGN["iname"]=sess.get_inputs()[0].name
        FOREIGN["isize"]=sess.get_inputs()[0].shape[-1] if isinstance(sess.get_inputs()[0].shape[-1], int) else 448
        d = np.load(ensure_file("foreign_ref.npz"), allow_pickle=True)
        embs = d["embeddings"].astype(np.float16)
        meta = json.load(open(ensure_file("foreign_ref_meta.json"), encoding="utf-8"))
        norms = np.linalg.norm(embs.astype(np.float32), axis=1)+1e-9
        ni={}
        for i,m in enumerate(meta):
            ni.setdefault(_digits(m.get("number")),[]).append(i)
        FOREIGN.update({"loaded":True,"embs":embs,"meta":meta,"norms":norms,"num_index":ni})
        print(f"Foreign catalogue loaded: {len(meta)} refs.")
    except Exception as e:
        print(f"Foreign load failed: {e}")

def fp(img):
    x = preprocess(img).unsqueeze(0).to("cpu")
    with torch.no_grad():
        v = model.encode_image(x); v = v/v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()[0]

def preprocess_for_hash(img):
    try:
        gray=img.convert("L"); bw=ImageOps.autocontrast(gray)
        bbox=bw.point(lambda p:255 if p<245 else 0).getbbox()
        if bbox:
            l,t,r,b=bbox; w,h=r-l,b-t
            if w>20 and h>20:
                px,py=int(w*.02),int(h*.02); img=img.crop((l+px,t+py,r-px,b-py))
    except Exception: pass
    return img.convert("RGB").resize((256,256))
def hash_match(img, hashes, topn=5):
    p=preprocess_for_hash(img)
    tp,ta,td,tw=imagehash.phash(p),imagehash.average_hash(p),imagehash.dhash(p),imagehash.whash(p)
    s=[((tp-e["p"])+(ta-e["a"])+(td-e["d"])+(tw-e["w"]),e) for e in hashes]
    s.sort(key=lambda x:x[0]); return s[:topn]

app = FastAPI()

@app.get("/")
def health():
    return {"status":"ok","games":{g:{"cards":len(v["meta"]),"hashes":len(v.get("hashes",[]))} for g,v in INDEX.items()},"total":total,"foreign_loaded":FOREIGN["loaded"]}

@app.post("/identify")
async def identify(file: UploadFile = File(...), game: str = Form(...)):
    if game not in INDEX:
        return {"error":f"unknown game '{game}'","known":list(INDEX.keys())}
    data=INDEX[game]
    img=Image.open(io.BytesIO(await file.read())).convert("RGB")
    q=fp(img); sims=data["vectors"]@q
    order=sims.argsort()[::-1][:5]; m=data["meta"]
    ct=float(sims[order[0]]) if len(order) else 0.0
    clip=[{"name":m[i]["name"],"number":m[i].get("number"),"set":m[i].get("set"),"score":float(sims[i]),"source":"clip"} for i in order]
    hm=[]
    if data.get("hashes"):
        for dist,e in hash_match(img,data["hashes"],5):
            hm.append({"name":e["name"],"number":e["number"],"set":e["set"],"distance":int(dist),"score":max(0.0,1.0-dist/120.0),"source":"hash"})
    best=clip[0] if clip else None
    if hm and ct<0.75 and hm[0]["score"]>0.80: best=hm[0]
    return {"game":game,"best":best,"clip":clip,"hash":hm}

def _score_pool(q, qn, idxs):
    embs=FOREIGN["embs"]; norms=FOREIGN["norms"]; meta=FOREIGN["meta"]
    sims=[(float(np.dot(embs[i].astype(np.float32),q)/(norms[i]*qn)),i) for i in idxs]
    sims.sort(reverse=True)
    return [{"name":meta[i].get("name"),"set":meta[i].get("set"),"number":meta[i].get("number"),"score":round(s,4)} for s,i in sims[:5]]

@app.post("/identify_foreign")
async def identify_foreign(file: UploadFile = File(...), number: str = Form(default="")):
    load_foreign()
    if not FOREIGN["loaded"]:
        return {"best":None,"confident":False,"error":"foreign unavailable"}
    img=Image.open(io.BytesIO(await file.read())).convert("RGB")
    q=_embed_foreign(img); qn=np.linalg.norm(q)+1e-9
    n=len(FOREIGN["meta"])

    # STAGE 1: narrow by the OCR-read number (works when JP/EN numbers match).
    tgt=_digits(number)
    idxs=FOREIGN["num_index"].get(tgt) if tgt else None
    if idxs:
        top=_score_pool(q, qn, idxs)
        best=top[0] if top else None
        if best:
            gap=(top[0]["score"]-top[1]["score"]) if len(top)>1 else 0.2
            if best["score"]>=0.32 and gap>=0.03:
                return {"best":best,"confident":True,"narrowed":True,"stage":"number","matches":top}

    # STAGE 2: number narrow missed (JP number != EN catalogue number). Full artwork search.
    # Higher bar since there's no number filter: needs a clear win to be trusted.
    top=_score_pool(q, qn, list(range(n)))
    best=top[0] if top else None
    conf=False
    if best:
        gap=(top[0]["score"]-top[1]["score"]) if len(top)>1 else 0.2
        conf = best["score"]>=0.40 and gap>=0.05
    return {"best":best if conf else None,"confident":conf,"narrowed":False,"stage":"full","matches":top}
