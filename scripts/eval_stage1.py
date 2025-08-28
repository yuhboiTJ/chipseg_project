import os, cv2, json, numpy as np, tensorflow as tf
from pathlib import Path

IMG_SIZE=384; MODEL="models/stage1_best.h5"; SPLIT="val"

def preprocess(img):
    h,w=img.shape[:2]; s=min(IMG_SIZE/h, IMG_SIZE/w); nh,nw=int(h*s),int(w*s)
    pad=np.zeros((IMG_SIZE,IMG_SIZE,3),np.uint8); pad[:nh,:nw]=cv2.resize(img,(nw,nh))
    return pad.astype(np.float32)/255.0,(h,w),(nh,nw)

def restore(m,hw,nh,nw): return cv2.resize(m[:nh,:nw],(hw[1],hw[0]),cv2.INTER_NEAREST)

def dice(gt,pr,eps=1e-6): inter=(gt&pr).sum(); return (2*inter+eps)/(gt.sum()+pr.sum()+eps)
def iou(gt,pr,eps=1e-6): inter=(gt&pr).sum(); return (inter+eps)/(gt.sum()+pr.sum()-inter+eps)

def pairs(split):
    d=Path(f"data/stage1/{split}")
    for p in sorted((d/"images").glob("*.png")):
        m=(d/"masks"/p.name.replace(".png","_mask.png"))
        if m.exists(): yield p,m

model=tf.keras.models.load_model(MODEL, compile=False)
ths=np.linspace(0.3,0.7,9); stats=[]

for t in ths:
    dices, ious, by_bg = [], [], {}
    for ip,mp in pairs(SPLIT):
        img=cv2.imread(str(ip)); gt=cv2.imread(str(mp),0)>127
        x,hw,sz=preprocess(img); y=model.predict(x[None,...],verbose=0)[0,...,0]
        pr=restore((y>t).astype(np.uint8)*255, hw, *sz)>127
        di,io=dice(gt,pr), iou(gt,pr); dices.append(di); ious.append(io)
        for tok in ip.stem.split("_"):
            if tok.lower().startswith("bg"):
                try:
                    bg=int(tok[2:]); by_bg.setdefault(bg, []).append(io)
                except: pass
    stats.append((float(t), float(np.mean(dices)), float(np.mean(ious)),
                 {int(k): float(np.mean(v)) for k,v in by_bg.items()}))

best=max(stats,key=lambda x:x[2])
print("Chosen threshold:", best[0])
for bg,miou in best[3].items(): print(f"Bg {bg}: {miou:.4f}")

Path("models").mkdir(exist_ok=True)
json.dump({"split":SPLIT,"results":stats,"best_threshold":best[0]},
          open("models/stage1_eval.json","w"), indent=2)
print("Wrote models/stage1_eval.json")
