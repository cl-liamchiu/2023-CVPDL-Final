# Inversion-Based Style Transfer with Diffusion Models

## Getting Started

### Prerequisites

For packages, see environment.yaml.

  ```sh
  conda env create -f environment.yaml
  conda activate ldm
  ```

### Downloading Pretrained Models and Embeddings

  ```sh
  bash download.sh
  ```
If not working, this is the [link](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt) to the pretrained models, and this is the [link](https://drive.google.com/drive/folders/1vte8eIp1QG9sQ4iKVeuQnB-RqmqMxVoD) to the pretrained embeddings.

### Folder tree
```
.
├── README.md
├── inference.ipynb
├── download.sh
├── sd-v1-4.ckpt
├── styles
│   ├── andre-derain_embeddings.pt
│   ├── longhair_embeddings.pt

```
### Inference
* Check the inference.ipynb
