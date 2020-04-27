# wgan-svhn
Wasserstein GAN for generating street view house numbers.

## Requirements:
numpy >= 1.17.4  
scipy >= 1.3.3  
torch >= 1.31  
torchvision >= 0.42  
pillow >= 4.1.1  

## Instructions:
mkdir -p gan_data/model gan_data/samples  
python wgan.py  

Model is trained for 100 epochs with final model at gan_data/model/wgan.pt, with samples generated per epoch in gan_data/samples.  
