from inst import inst
import argparse

parser = argparse.ArgumentParser()

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

if __name__ == '__main__':
    parser.add_argument("input_path", help="path to sketch image", type=str)
    parser.add_argument("output_path", help="path to output image", type=str)
    parser.add_argument("--style", help="enter a new style", type=str)
    args = parser.parse_args()

    if args.style is None:
        exit("Please enter a style")

    inst_style_list = ["modern", "longhair", "andre-derain", "woman"]
    if args.style not in inst_style_list:
        print("style list: ", inst_style_list)
        exit("Please enter a valid style")
    
    inst_result = generate_inst_image(args.style, args.input_path, "*")
    inst_result = inst_result.resize((256, 256))
    inst_result.save(args.output_path)
