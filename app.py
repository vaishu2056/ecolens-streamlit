import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")  # Replace with your model file

model = load_model()

# Define label colors
CLASS_LABELS = {
    0: ("Unlabeled", [0, 0, 0]),
    1: ("Road", [128, 64, 128]),
    2: ("Sidewalk", [244, 35, 232]),
    3: ("Building", [70, 70, 70]),
    4: ("Wall", [102, 102, 156]),
    5: ("Fence", [190, 153, 153]),
    6: ("Pole", [153, 153, 153]),
    7: ("Vegetation", [107, 142, 35]),
    8: ("Car", [0, 0, 142]),
}

def create_colored_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, (label, color) in CLASS_LABELS.items():
        color_mask[mask == class_id] = color
    return color_mask

def preprocess_image(img, target_size=(256, 256)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_mask(image):
    preprocessed = preprocess_image(image)
    pred = model.predict(preprocessed)[0]
    mask = np.argmax(pred, axis=-1)
    return mask

# Streamlit UI
st.title("Semantic Segmentation UI - Ecolens")

uploaded_file = st.file_uploader("Upload a Satellite Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        mask = predict_mask(image)
        color_mask = create_colored_mask(mask)

    st.image(color_mask, caption="Predicted Mask with Labels", use_column_width=True)
