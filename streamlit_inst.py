import streamlit as st
from PIL import Image
import numpy as np

from inst import inst

@st.cache_data
def generate_inst_image(style, image_path, prompt):
    # style: woman, modern, longhair, andre-derain
    style_image = f'image_style_transfer/styles/{style}.jpg'
    print("style image: ", style_image)

    return inst(prompt = prompt, \
     content_dir = image_path, \
     style_dir = f'{style_image}', \
     ddim_steps = 70, \
     strength = 0.7, \
     seed=42, \
     style=style)
    
st.title("Inst Style Transfer")

# 1. upload image
# 2. select style
# 3. generate image

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert('RGB')

    # resize to 512*512 and show
    original_image = original_image.resize((512, 512))
    original_image = np.array(original_image)
    st.image(original_image, caption="Uploaded Image", use_column_width="auto")

prompt = st.text_input("Prompt", "")

inst_style_list = ["Default", "modern", "longhair", "andre-derain", "woman"]

style = st.selectbox("Select a style", inst_style_list)
if style == "Default":
    style = None
if ((uploaded_file is not None) and (style is not None)):
    if prompt is None or prompt == "":
        prompt = "*"
    inst_result = generate_inst_image(style, uploaded_file, prompt)
    st.image([np.array(inst_result)], caption=["Generated Image"], use_column_width="auto")
