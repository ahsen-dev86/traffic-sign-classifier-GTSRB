import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image

# ==== CONFIG ====
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered"
)

# ==== HEADER ====
st.title("🚦 Traffic Sign Classification App")
st.write("""
Upload one or more traffic sign images and see how well the model can recognize them.  
This app is built on the **GTSRB dataset** and trained with a deep learning model.  
Developed by **Muhammad Ahsan** ✨  

✅ Works on both **laptop** and **mobile**.  
""")

# ==== LOAD MODEL + LABELS ====
MODEL_PATH = "traffic_sign_model.h5"   # make sure file is in same folder
LABELS_PATH = "label_names.csv"        # CSV with ClassId, SignName

@st.cache_resource
def load_all():
    model = load_model(MODEL_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    label_names = dict(zip(labels_df["ClassId"], labels_df["SignName"]))
    return model, label_names

model, label_names = load_all()

# ==== HELPERS ====
def show_topk(probs, k=3):
    idxs = probs.argsort()[-k:][::-1]
    return [(label_names.get(i, i), float(probs[i])) for i in idxs]

def prep_rgb(rgb):
    im = cv2.resize(rgb, (32, 32)).astype("float32") / 255.0
    return np.expand_dims(im, 0)

def predict_image(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    p = model.predict(prep_rgb(rgb), verbose=0)[0]
    top3 = show_topk(p, k=3)
    return top3

# ==== UPLOAD ====
uploaded_files = st.file_uploader(
    "Upload traffic sign images", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# ==== PROCESS ====
if uploaded_files:
    results = []
    for file in uploaded_files:
        image = Image.open(file)
        top3 = predict_image(image)

        # Display image + prediction
        st.image(image, caption=f"Uploaded: {file.name}", use_container_width=True)
        st.markdown(f"**Prediction:** {top3[0][0]} ({top3[0][1]*100:.2f}%)")

        # Show Top-3
        with st.expander("🔎 See Top-3 Predictions"):
            for name, prob in top3:
                st.write(f"- {name}: {prob*100:.2f}%")

        results.append({
            "File": file.name,
            "Prediction": top3[0][0],
            "Confidence": f"{top3[0][1]*100:.2f}%",
            "Top-3": ", ".join([f"{n} ({p*100:.1f}%)" for n,p in top3])
        })

    # Show summary table
    df = pd.DataFrame(results)
    st.subheader("📊 Summary of Predictions")
    st.dataframe(df, use_container_width=True)
