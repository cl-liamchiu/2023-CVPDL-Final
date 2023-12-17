from clip import fast_clip
import argparse

parser = argparse.ArgumentParser()

if __name__ == '__main__':
    parser.add_argument("input_path", help="path to sketch image", type=str)
    parser.add_argument("output_path", help="path to output image", type=str)
    parser.add_argument("--style", help="enter a new style", type=str)
    args = parser.parse_args()

    if args.style is None:
        exit("Please enter a style")
    
    
    