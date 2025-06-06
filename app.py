import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Class names and colors
CLASS_NAMES = ["Background clutter", "Building", "Road", "Tree", "Small Ground"]
COLORS = [
    (128, 128, 128),  # Grey
    (255, 0, 0),      # Red
    (255, 255, 0),    # Yellow
    (0, 255, 0),      # Green
    (255, 0, 255)     # Purple
]

def create_color_mask(pred_mask):
    color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for class_index, color in enumerate(COLORS):
        color_mask[pred_mask == class_index] = color
    return color_mask

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

    # Predict
    pred = model.predict(img_array)
    pred_mask = np.argmax(pred[0], axis=-1)
    pred_mask_resized = tf.image.resize(pred_mask[..., np.newaxis], image.size[::-1], method="nearest")
    pred_mask_resized = tf.squeeze(pred_mask_resized).numpy().astype(np.uint8)

    # Generate color mask
    color_mask = create_color_mask(pred_mask_resized)
    color_mask_resized = Image.fromarray(color_mask).resize(image.size, resample=Image.NEAREST)

    # Overlay
    overlay = Image.blend(image.convert("RGBA"), color_mask_resized.convert("RGBA"), alpha=0.5)

    # Display image and legend side by side
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### Legend")
        for name, color in zip(CLASS_NAMES, COLORS):
            hex_color = '#%02x%02x%02x' % color
            st.markdown(f"<span style='color:{hex_color}; font-size:16px;'>⬛ {name}</span>", unsafe_allow_html=True)
    with col2:
        st.image(overlay, caption="Overlay on Original Image", use_column_width=True)

    # Pie chart of class distribution
    st.markdown("### Class Distribution (Pie Chart)")
    class_counts = [np.sum(pred_mask_resized == i) for i in range(len(CLASS_NAMES))]
    fig, ax = plt.subplots()
    ax.pie(class_counts, labels=CLASS_NAMES, colors=np.array(COLORS)/255.0, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
