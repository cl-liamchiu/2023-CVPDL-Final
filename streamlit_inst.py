import streamlit as st
import numpy as np

import os

from masksketch.utils import read_image_from_path
from category import category_list

from inst import inst

# 1. step 1: upload sketch image and select category
# 2. step 2: use masksketch to generate image and save to temp file path
# 3. step 3: use clipstyler to generate image and save as output.jpg

# download masksketch model
os.makedirs("checkpoints", exist_ok=True)


# 1. step 1: upload sketch image and select category
st.title("Sketch to Captivating Image")
st.write("Upload a sketch and select a category to generate an image")
cat = st.selectbox("Select a category", category_list)
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

print("selected category: ", cat)
print("uploaded file: ", uploaded_file)


# get select category index
cat_index = category_list.index(cat)

output_path = "outputs/sketch_51.jpg"
if (uploaded_file is not None):
    image = read_image_from_path(
        uploaded_file,
        height=256,
        width=256)
    st.image([image], caption=['Uploaded Image'], use_column_width="auto")

if ((uploaded_file is not None) and (cat is not None)):
    sketch_result = read_image_from_path(
        output_path,
        height=256,
        width=256)

    sketch_result_t = sketch_result.resize((512, 512))
else:
    sketch_result = None
    sketch_result_t = None


@st.cache_data
def generate_inst_image(style, image_path, prompt):
    # style: woman, modern, longhair, andre-derain
    style_image = f'image_style_transfer/styles/{style}.jpg'
    print("style image: ", style_image)

    return inst(prompt = prompt, \
     content_dir = image_path, \
     style_dir = f'{style_image}', \
     ddim_steps = 70, \
     strength = 0.3, \
     seed=42, \
     style=style)
    
tab1, tab2 = st.tabs(["CLIPstyler", "InST"])

with tab1:
    clip_style_list = ["None", "acrylic", "desert_sand", "inkwash_painting", "oil_bluered_brush",
        "stonewall", "water_purple_brush", "anime",
        "blue_wool", "cyberpunk", "mondrian", "papyrus", "Custom"]

    style = st.selectbox("Select a style", clip_style_list)
    if style == "Custom":
        style = st.text_input("Custom Style", "anime")
        style = "sketch_blackpencil"

with tab2:
    inst_style_list = ["Default", "modern", "longhair", "andre-derain", "woman"]

    inst_style = st.selectbox("Select a style", inst_style_list)
    if inst_style == "Default":
        inst_style = None
    
    if inst_style is not None:
        style_image_path =  "image_style_transfer/styles/" + inst_style + ".jpg"
        style_image = read_image_from_path(
                style_image_path,
                height=256,
                width=256)
        st.image([style_image], caption=['Reference Image'], use_column_width="auto")

if st.button('Generate Image'):
    if ((output_path is not None) and (inst_style is not None)):
        prompt = "*"
        print("output path: ", output_path)
        inst_result = generate_inst_image(inst_style, output_path, prompt)
        inst_result = inst_result.resize((256, 256))
        sketch_result = read_image_from_path(
            output_path,
            height=256,
            width=256)
        st.image([sketch_result, inst_result], caption=["Sketch to image", "Style Image"], use_column_width="auto")


