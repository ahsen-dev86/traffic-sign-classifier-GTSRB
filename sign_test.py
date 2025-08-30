# --- Upload & Predict on MULTIPLE traffic sign images (Colab) ---
from google.colab import files
import cv2, numpy as np, matplotlib.pyplot as plt, pandas as pd
from tensorflow.keras.models import load_model

# ==== 1) Model + labels ====
MODEL_PATH  = "/content/drive/MyDrive/GTSRB/traffic_sign_model.h5"
LABELS_PATH = "/content/drive/MyDrive/GTSRB/label_names.csv"

model = load_model(MODEL_PATH)
labels_df = pd.read_csv(LABELS_PATH)
label_names = dict(zip(labels_df["ClassId"], labels_df["SignName"]))

# ==== 2) Upload multiple ====
uploaded = files.upload()  # select multiple images here
image_paths = list(uploaded.keys())

# ==== 3) Helpers ====
def show_topk(probs, k=3):
    idxs = probs.argsort()[-k:][::-1]
    return [(label_names.get(i,i), float(probs[i])) for i in idxs]

def crop_by_color(rgb, color="red"):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if color == "red":
        mask1 = cv2.inRange(hsv, (0,70,50), (10,255,255))
        mask2 = cv2.inRange(hsv, (170,70,50), (180,255,255))
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "blue":
        mask = cv2.inRange(hsv, (100,100,50), (140,255,255))
    elif color == "yellow":
        mask = cv2.inRange(hsv, (20,100,100), (40,255,255))
    else:
        return rgb, None

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return rgb, None
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    pad = int(0.15 * max(w,h))
    x0,y0 = max(0,x-pad), max(0,y-pad)
    x1,y1 = min(rgb.shape[1],x+w+pad), min(rgb.shape[0],y+h+pad)
    return rgb[y0:y1, x0:x1], (x0,y0,x1,y1)

def prep_rgb(rgb):
    im = cv2.resize(rgb, (32,32)).astype("float32") / 255.0
    return np.expand_dims(im,0)

# ==== 4) Loop over all uploaded images ====
results_table = []

for path in image_paths:
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # build candidate crops
    crops = {"full": rgb}
    for c in ["red","blue","yellow"]:
        crop, box = crop_by_color(rgb, c)
        if box: crops[f"{c}_crop"] = crop

    # predict on each
    best_probs = None
    best_conf = -1
    best_name = None
    for variant, img in crops.items():
        p = model.predict(prep_rgb(img), verbose=0)[0]
        if p.max() > best_conf:
            best_conf = p.max()
            best_probs = p
            best_name = variant

    # summarize
    top3 = show_topk(best_probs, k=3)
    results_table.append({
        "File": path,
        "Best variant": best_name,
        "Prediction": top3[0][0],
        "Confidence": f"{top3[0][1]*100:.2f}%",
        "Top-3": ", ".join([f"{n} ({p*100:.1f}%)" for n,p in top3])
    })

    # show quick visualization
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(f"{path}\nPred: {top3[0][0]} ({top3[0][1]*100:.2f}%)")
    plt.show()

# ==== 5) Print summary table ====
import pandas as pd
df = pd.DataFrame(results_table)
print(df)
