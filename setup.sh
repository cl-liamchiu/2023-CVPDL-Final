conda install --yes -c pytorch pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
pip install -r requirements.txt
pip install jax[cuda12_pip]==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

cd image_style_transfer
bash download.sh
cd ..

