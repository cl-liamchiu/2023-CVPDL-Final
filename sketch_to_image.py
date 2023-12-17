import argparse
from category import category_list
from PIL import Image
import numpy as np

import jax
import os
import itertools

from masksketch.utils import read_image_from_path
from masksketch.sketch_conditional_inference import MaskSketch_generator
from masksketch.configs import masksketch_class_cond_config

os.makedirs("checkpoints", exist_ok=True)
models_to_download = itertools.product(*[ ["maskgit", "tokenizer"],   [256, ] ])

def generate_image(image_path, cat_index, output_path):

    for (type_, resolution) in models_to_download:
        canonical_path = MaskSketch_generator.checkpoint_canonical_path(type_, resolution)
        if os.path.isfile(canonical_path):
            print(f"Checkpoint for {resolution} {type_} already exists, not downloading again")
        else:
            source_url = f'https://storage.googleapis.com/maskgit-public/checkpoints/{type_}_imagenet{resolution}_checkpoint'
            os.system(f"wget {source_url} -O {canonical_path}")

    image = read_image_from_path(
        image_path,
        height=256,
        width=256)

    config = masksketch_class_cond_config.get_config()
    generator_256 = MaskSketch_generator(image_size=256, config=config)
    arbitrary_seed = 42
    rng = jax.random.PRNGKey(arbitrary_seed)
    rng, sample_rng = jax.random.split(rng)

    sampled_images = generator_256.generate_samples(rng=sample_rng, input_image=image, class_label=cat_index, num_iterations=500)
    generated_image = sampled_images[0] * 255.0

    arr = np.round(np.array(generated_image))
    # if values are outside [0, 255], clip them
    arr = np.clip(arr, 0, 255)
    # convert to uint8
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    
    im.save(output_path)

parser = argparse.ArgumentParser()

if __name__ == '__main__':
    parser.add_argument("input_path", help="path to sketch image", type=str)
    parser.add_argument("output_path", help="path to output image", type=str)
    parser.add_argument("--show_category", help="show category", default=False, type=bool)
    parser.add_argument("--category_number", help="category", type=int)
    args = parser.parse_args()

    if args.show_category or args.category_number is None:
        print("category list: ")
        for i, cat in enumerate(category_list):
            print(f"{i}: {cat}")
        
        # enter category number
        category_number = input("Enter category number: ")
    else:
        category_number = args.category_number
    
    generate_image(args.input_path, category_number, args.output_path)




