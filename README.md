# 2023-CVPDL-Final

## Environments

- OS: Ubuntu 18.04
- Python Version: 3.8
- GPU: NVIDIA RTX A6000
- GPU RAM: 48 GB
- Cuda Version: 11.3
- PyTorch Version: 1.7.1

## Installation

```sh
conda create --yes -n cvpdl-final python=3.8
conda activate cvpdl-final
conda install --yes -c pytorch pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
pip install -r requirements.txt
pip install jax[cuda12_pip]==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Download model of InST

```sh
cd image_style_transfer
bash download.sh
```

### Download model of CLIPstyler

- Download DIV2K dataset [LINK](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

- To train the model, please download the pre-trained vgg encoder & decoder models in [LINK](https://drive.google.com/drive/folders/17UDzXtp9IZlerFjGly3QEm2uU3yi7siO?usp=sharing).

- Please save the downloaded models in `./models` directory

## Run Streamlit

```sh
streamlit run streamlit.py
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
