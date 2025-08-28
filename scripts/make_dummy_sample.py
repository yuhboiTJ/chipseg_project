import cv2, numpy as np, pathlib as P, shutil

root = P.Path("data/stage1")
for split in ["train","val","test"]:
    (root/split/"images").mkdir(parents=True, exist_ok=True)
    (root/split/"masks").mkdir(parents=True, exist_ok=True)

# simple polygon "chip" on a colored background
img = np.full((512,512,3), (40,20,200), np.uint8)
pts = np.array([[120,80],[420,160],[380,420],[80,380]], np.int32)
cv2.fillPoly(img, [pts], (10,10,10))

mask = np.zeros((512,512), np.uint8)
cv2.fillPoly(mask, [pts], 255)

name = "C1_Bg1_Z1.png"
(cv2.imwrite(str(root/"train"/"images"/name), img))
(cv2.imwrite(str(root/"train"/"masks"/name.replace(".png","_mask.png")), mask))

# copy to val/test so scripts have something to run
for split in ["val","test"]:
    shutil.copy(root/"train"/"images"/name, root/split/"images"/name)
    shutil.copy(root/"train"/"masks"/name.replace(".png","_mask.png"),
                root/split/"masks"/name.replace(".png","_mask.png"))

print("Dummy sample written in data/stage1/{train,val,test}/{images,masks}")
