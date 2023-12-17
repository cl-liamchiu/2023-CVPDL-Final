# 2023-CVPDL-Final

## Installation

```sh
conda create --yes -n cvpdl-final python=3.8
conda activate cvpdl-final
conda install --yes -c pytorch pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
pip install -r requirements.txt
pip install jax[cuda12_pip]==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## MaskSketch

[MaskSketch](https://github.com/google-research/masksketch) is a structure-conditional image generation model based on [MaskGIT](https://github.com/google-research/maskgit). Our method leverages the structure-preserving properties of the self-attention maps of MaskGIT to generate realistic images that follow the structure given an input image or sketch.

```sh
python sketch_to_image.py [--show_category SHOW_CATEGORY] [--category_number CATEGORY_NUMBER] input_path output_path
# example
python sketch_to_image.py --category_number=35 ./imgs/turtle_2.jpg test.jpg
```

## CLIPstyler

### Fast Style Transfer

```sh
python clip_style_fast_style_transfer.py --style STYLE input_path output_path
#example
python clip_style_fast_style_transfer.py --style="anime" ./test.jpg ./test_clip_fast.jpg
```

## InST (Inversion-Based Style Transfer with Diffusion Models)

### Single Style Transfer

```sh
python inst_single_style_transfer.py --style STYLE input_path output_path
# example
python inst_single_style_transfer.py --style="modern" ./test.jpg ./test_inst_single.jpg
```
