import streamlit as st
from PIL import Image
import numpy as np

import jax
import os
import itertools

from masksketch.utils import read_image_from_path
from masksketch.sketch_conditional_inference import MaskSketch_generator
from masksketch.configs import masksketch_class_cond_config
from category import category_list


from clip import fast_clip
# from inst import inst

# 1. step 1: upload sketch image and select category
# 2. step 2: use masksketch to generate image and save to temp file path
# 3. step 3: use clipstyler to generate image and save as output.jpg

# download masksketch model
os.makedirs("checkpoints", exist_ok=True)
models_to_download = itertools.product(*[ ["maskgit", "tokenizer"],   [256, ] ])

for (type_, resolution) in models_to_download:
    canonical_path = MaskSketch_generator.checkpoint_canonical_path(type_, resolution)
    if os.path.isfile(canonical_path):
        print(f"Checkpoint for {resolution} {type_} already exists, not downloading again")
    else:
        source_url = f'https://storage.googleapis.com/maskgit-public/checkpoints/{type_}_imagenet{resolution}_checkpoint'
        os.system(f"wget {source_url} -O {canonical_path}")

# 1. step 1: upload sketch image and select category
st.title("Sketch to Style Image")
st.write("Upload a sketch and select a category to generate an image")
cat = st.selectbox("Select a category", category_list)
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

print("selected category: ", cat)
print("uploaded file: ", uploaded_file)

# get select category index
cat_index = category_list.index(cat)

if (uploaded_file is not None):
    image = read_image_from_path(
        uploaded_file,
        height=256,
        width=256)
    st.image([image], caption=['Uploaded Image'], use_column_width="auto")


@st.cache_data
def generate_image(image, cat_index):
    config = masksketch_class_cond_config.get_config()
    generator_256 = MaskSketch_generator(image_size=256, config=config)
    arbitrary_seed = 42
    rng = jax.random.PRNGKey(arbitrary_seed)

    run_mode = 'normal'
    p_generate_256_samples = generator_256.p_generate_samples()

    rng, sample_rng = jax.random.split(rng)

    sampled_images = generator_256.generate_samples(rng=sample_rng, input_image=image, class_label=cat_index, num_iterations=500)
    generated_image = sampled_images[0] * 255.0

    arr = np.round(np.array(generated_image))
    # if values are outside [0, 255], clip them
    arr = np.clip(arr, 0, 255)
    # convert to uint8
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    
    # save image to outputs folder
    os.makedirs("outputs", exist_ok=True)
    file_len = len(os.listdir("outputs"))
    im.save(f"outputs/sketch_{file_len}.jpg")
    return im, f"outputs/sketch_{file_len}.jpg"

@st.cache_data
def generate_clip_image(style, image):
    file_len = len(os.listdir("outputs"))
    img = fast_clip(image, text_cond=style, output_path=f"outputs/clip_{style}_{file_len}.jpg")
    img = img * 255.0
    arr = np.round(np.array(img))
    # if values are outside [0, 255], clip them
    arr = np.clip(arr, 0, 255)
    # convert to uint8
    arr = arr.astype(np.uint8)
    print("arr shape: ", arr.shape)
    # transform from (3,512,512) to (512,512,3)
    arr = np.transpose(arr, (1,2,0))
    im = Image.fromarray(arr, mode="RGB")

    # resize to 256x256
    im = im.resize((256, 256))
    return im

# @st.cache_data
# def generate_inst_image(style, image_path, prompt):
#     # style: woman, modern, longhair, andre-derain
#     style_image = f'image_style_transfer/styles/{style}.jpg'
#     print("style image: ", style_image)

#     return inst(prompt = prompt, \
#      content_dir = image_path, \
#      style_dir = f'{style_image}', \
#      ddim_steps = 70, \
#      strength = 0.7, \
#      seed=42, \
#      style=style)


tab1, tab2 = st.tabs(["CLIPstyler", "InST"])

with tab1:
    clip_style_list = ["None", "acrylic", "desert_sand", "inkwash_painting", "oil_bluered_brush",
        "stonewall", "water_purple_brush", "anime",
        "blue_wool", "cyberpunk", "mondrian", "papyrus", "Custom"]

    style = st.selectbox("Select a style", clip_style_list)
    if style == "Custom":
        style = st.text_input("Custom Style", "")
        if style != "":
            style = "sketch_blackpencil"
        else:
            style = None

with tab2:
    inst_style_list = ["Default", "modern", "longhair", "andre-derain", "woman"]

    inst_style = st.selectbox("Select a style", inst_style_list)
    if inst_style == "Default":
        inst_style = None
    # if ((output_path is not None) and (inst_style is not None)):
    #     prompt = "*"
    #     print("output path: ", output_path)
    #     inst_result = generate_inst_image(inst_style, output_path, prompt)
    #     st.image([np.array(inst_result)], caption=["Generated Image"], use_column_width="auto


if st.button('Generate Image'):
    output_path = None
    if ((uploaded_file is not None) and (cat is not None)):
        sketch_result, output_path = generate_image(image, cat_index)

        print("shape: ", np.array(sketch_result).shape) # (256,256,3)

        # upsampling to 512x512
        sketch_result_t = sketch_result.resize((512, 512))
    else:
        sketch_result = None
        sketch_result_t = None


    if ((sketch_result_t is not None) and (style is not None) and (style != "None")):
        # print("sketch result: ", sketch_result_t)
        clip_result = generate_clip_image(style, sketch_result_t)
        print("sketch result: ", sketch_result_t)
        print("clip result: ", clip_result)
        st.image([sketch_result, np.array(clip_result) ], caption=['Sketch to Image', "Style Image"], use_column_width="auto")
