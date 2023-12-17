from clip import fast_clip
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()

def generate_clip_image(style, image_path, output_path):
    image = Image.open(image_path)
    image = image.resize((512, 512))
    img = fast_clip(image, text_cond=style, output_path=output_path)
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
    im.save(output_path)


if __name__ == '__main__':
    parser.add_argument("input_path", help="path to sketch image", type=str)
    parser.add_argument("output_path", help="path to output image", type=str)
    parser.add_argument("--style", help="choose a style from list", type=str)
    args = parser.parse_args()

    if args.style is None:
        exit("Please enter a style")
    
    clip_style_list = ["None", "acrylic", "desert_sand", "inkwash_painting", "oil_bluered_brush",
        "stonewall", "water_purple_brush", "anime",
        "blue_wool", "cyberpunk", "mondrian", "papyrus", "Custom"]
    
    if args.style not in clip_style_list:
        print("style list: ", clip_style_list)
        exit("Please enter a valid style")

    generate_clip_image(args.style, args.input_path, args.output_path)