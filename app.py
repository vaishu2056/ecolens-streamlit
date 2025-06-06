import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Class names and corresponding colors (adjust according to your dataset)
CLASS_NAMES = ["Background clutter", "Building", "Road", "Tree", "Small Land"]
COLORS = [
    (128, 128, 128),  # Background clutter (Grey)
    (255, 0, 0),      # Building
    (255, 255, 0),    # Road
    (0, 255, 0),      # Tree
    (255, 0, 255)     # Small Land
]

# Function to create a color mask from class indices
def create_color_mask(pred_mask):
    color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for class_index, color in enumerate(COLORS):
        color_mask[pred_mask == class_index] = color
    return color_mask

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras", compile=False)

model = load_model()

st.title("🌱 EcoLens: Tree and Land Segmentation")

# Upload image
uploaded_file = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict segmentation
    pred = model.predict(img_array)
    pred_mask = np.argmax(pred[0], axis=-1)

    # Resize predicted mask back to original image size
    pred_mask_resized = tf.image.resize(pred_mask[..., np.newaxis], image.size[::-1], method="nearest")
    pred_mask_resized = tf.squeeze(pred_mask_resized).numpy().astype(np.uint8)

    # Create color mask and resize it to original size
    color_mask = create_color_mask(pred_mask_resized)
    color_mask_resized = Image.fromarray(color_mask).resize(image.size, resample=Image.NEAREST)

    # Show color mask
    st.image(color_mask_resized, caption="Predicted Segmentation", use_column_width=True)

    # Overlay option
    st.markdown("### Overlay Segmentation")
    overlay = Image.blend(image.convert("RGBA"), color_mask_resized.convert("RGBA"), alpha=0.5)
    st.image(overlay, caption="Overlay on Original Image", use_column_width=True)

    # Show legend
    st.markdown("### Legend")
    for name, color in zip(CLASS_NAMES, COLORS):
        hex_color = '#%02x%02x%02x' % color
        st.markdown(f"<span style='color:{hex_color}; font-size:16px;'>⬛ {name}</span>", unsafe_allow_html=True)
