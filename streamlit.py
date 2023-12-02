import streamlit as st
from PIL import Image
import numpy as np

import jax
import os
import itertools

from masksketch.utils import visualize_images, read_image_from_path, restore_from_path, draw_image_with_bbox, Bbox
from masksketch.sketch_conditional_inference import MaskSketch_generator
from masksketch.configs import masksketch_class_cond_config
from category import category_list


from clip import fast_clip

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
st.title("Sketch to Stylish Image")
st.write("Upload a sketch and select a category to generate an image")
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

cat = st.selectbox("Select a category", category_list)
print("selected category: ", cat)
print("uploaded file: ", uploaded_file)


# get select category index
cat_index = category_list.index(cat)


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
    return im

if ((uploaded_file is not None) and (cat is not None)):
    image = read_image_from_path(
        uploaded_file,
        height=256,
        width=256)
    sketch_result = generate_image(image, cat_index)

    st.image([image, sketch_result], caption=['Uploaded Image', "Generated Image"], use_column_width="auto")

    # 2. step 2: use masksketch to generate image and save to temp file path
    # 3. step 3: use clipstyler to generate image and save as output.jpg

print("sketch result: ", sketch_result)
print("shape: ", np.array(sketch_result).shape) # (256,256,3)

# upsampling to 512x512
sketch_result_t = sketch_result.resize((512, 512))

@st.cache_data
def generate_clip_image(style, image):
    img = fast_clip(image, text_cond=style, output_path="./output.jpg")
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

style_list = ["acrylic", "desert_sand", "inkwash_painting", "oil_bluered_brush",
    "sketch_blackpencil", "stonewall", "water_purple_brush", "anime",
    "blue_wool", "cyberpunk", "mondrian", "papyrus"]

style = st.selectbox("Select a style", style_list)
if ((sketch_result_t is not None) and (style is not None)):
    # print("sketch result: ", sketch_result_t)
    clip_result = generate_clip_image(style, sketch_result_t)
    print("sketch result: ", sketch_result_t)
    print("clip result: ", clip_result)
    st.image([sketch_result, np.array(clip_result) ], caption=['Original Image', "Generated Image"], use_column_width="auto")

# a button to clear results
if st.button("Clear Results"):
    sketch_result = None
    uploaded_file = None
