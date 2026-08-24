import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Image Array Operations", page_icon="🖼️", layout="centered")
st.title("🖼️ Image Array Operations")

# ---------------------------------------------------------------------
# Input: editable pixel grid
# ---------------------------------------------------------------------
default_data = pd.DataFrame(
    [
        [45, 120, 200, 10],
        [80, 255, 150, 60],
        [30, 90, 180, 220],
        [0, 100, 210, 250],
    ],
    index=[f"Row {i}" for i in range(4)],
    columns=[f"Col {j}" for j in range(4)],
)

st.subheader("Pixel Values (0-255 grayscale)")
st.caption("Edit values directly. Rows/columns can be resized, but note some operations (rotation, cropping) assume a square-ish grid.")

edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

if edited_df.empty or edited_df.isnull().values.any():
    st.warning("Please make sure every cell has a numeric value.")
    st.stop()

image = np.clip(edited_df.to_numpy(dtype=float), 0, 255)


def show_image_pair(arr, caption, scale=40):
    """Render an array as a scaled-up grayscale image next to its raw values."""
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img_big = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    st.image(img_big, caption=caption, use_container_width=False)


# ---------------------------------------------------------------------
# Original image
# ---------------------------------------------------------------------
st.header("Original Image")
col1, col2 = st.columns([1, 1])
with col1:
    show_image_pair(image, "Original")
with col2:
    st.dataframe(pd.DataFrame(image.astype(int)), use_container_width=True)

# ---------------------------------------------------------------------
# Inverted / bright / dark
# ---------------------------------------------------------------------
st.header("Transformations")

inverted_image = 255 - image
bright_offset = st.slider("Brighten by", min_value=0, max_value=255, value=50)
bright_image = np.clip(image + bright_offset, 0, 255)
dark_offset = st.slider("Darken by", min_value=0, max_value=255, value=50)
dark_image = np.clip(image - dark_offset, 0, 255)

tab1, tab2, tab3 = st.tabs(["Inverted", "Brightened", "Darkened"])
with tab1:
    show_image_pair(inverted_image, "Inverted (255 - image)")
with tab2:
    show_image_pair(bright_image, f"Brightened (+{bright_offset}, clipped)")
with tab3:
    show_image_pair(dark_image, f"Darkened (-{dark_offset}, clipped)")

# ---------------------------------------------------------------------
# Black and white threshold
# ---------------------------------------------------------------------
st.header("Black & White Threshold")

bw_threshold = st.slider("Threshold", min_value=0, max_value=255, value=128)
black_and_white = np.where(image > bw_threshold, 255, 0)
show_image_pair(black_and_white, f"Black & White (threshold = {bw_threshold})")

# ---------------------------------------------------------------------
# Center crop
# ---------------------------------------------------------------------
st.header("Crop")

max_r, max_c = image.shape
col1, col2 = st.columns(2)
row_range = col1.slider("Row range", 0, max_r, (1, min(3, max_r)))
col_range = col2.slider("Column range", 0, max_c, (1, min(3, max_c)))

if row_range[1] > row_range[0] and col_range[1] > col_range[0]:
    crop = image[row_range[0]:row_range[1], col_range[0]:col_range[1]]
    col1, col2 = st.columns([1, 1])
    with col1:
        show_image_pair(crop, "Cropped region", scale=60)
    with col2:
        st.dataframe(pd.DataFrame(crop.astype(int)), use_container_width=True)
else:
    st.info("Pick a valid (non-empty) row and column range.")

# ---------------------------------------------------------------------
# Pixel stats
# ---------------------------------------------------------------------
st.header("Pixel Statistics")

dark_pixel_cutoff = st.number_input("Dark pixel cutoff (< value)", min_value=0, max_value=255, value=100)
bright_pixel_cutoff = st.number_input("Bright pixel cutoff (> value)", min_value=0, max_value=255, value=200)

dark_pixel_count = int(np.sum(image < dark_pixel_cutoff))
bright_pixel_count = int(np.sum(image > bright_pixel_cutoff))
avg_brightness = np.mean(image)

col1, col2, col3 = st.columns(3)
col1.metric(f"Pixels < {dark_pixel_cutoff}", dark_pixel_count)
col2.metric(f"Pixels > {bright_pixel_cutoff}", bright_pixel_count)
col3.metric("Average brightness", f"{avg_brightness:.2f}")

# ---------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------
st.header("Rotation")

rotate_k = st.select_slider("Rotate (× 90° counter-clockwise)", options=[1, 2, 3], value=1)
rotated = np.rot90(image, rotate_k)
show_image_pair(rotated, f"Rotated {rotate_k * 90}°")

# ---------------------------------------------------------------------
# Brightest pixel
# ---------------------------------------------------------------------
st.header("Brightest Pixel")

brightest_pixel = np.max(image)
brightest_position = np.unravel_index(np.argmax(image), image.shape)

col1, col2 = st.columns(2)
col1.metric("Brightest pixel value", f"{brightest_pixel:.0f}")
col2.metric("Position (row, col)", str(brightest_position))
