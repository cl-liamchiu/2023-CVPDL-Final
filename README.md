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
MaskSketch is a structure-conditional image generation model based on [MaskGIT](https://github.com/google-research/maskgit). Our method leverages the structure-preserving properties of the self-attention maps of MaskGIT to generate realistic images that follow the structure given an input image or sketch. 
