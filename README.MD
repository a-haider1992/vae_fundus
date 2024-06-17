# VAE for Fundus Image Reconstruction

## Introduction
In this project, we explore the use of a Variational Autoencoder (VAE) for fundus image reconstruction. Fundus images are widely used in ophthalmology to diagnose and monitor various eye diseases. However, capturing high-quality fundus images can be challenging, and there is often a need for image enhancement or reconstruction.

## Variational Autoencoder (VAE)
A Variational Autoencoder is a type of generative model that can learn to encode and decode complex data distributions. It consists of an encoder network that maps input data to a latent space, and a decoder network that reconstructs the original data from the latent space representation.

## Fundus Image Reconstruction
Fundus image reconstruction using a VAE involves training the VAE on a dataset of fundus images. The encoder network learns to encode the images into a lower-dimensional latent space representation, while the decoder network learns to reconstruct the original images from the latent space.

## Trained model
We use 1.7M fundus tiles (or patches) of size 3x256x256 with reconstruction SSIM value of 0.99. The trained model can be found at [a link](https://qubstudentcloud-my.sharepoint.com/:u:/r/personal/3057693_ads_qub_ac_uk/Documents/VAE_FUNDUS/model.pth?csf=1&web=1&e=fZRA3K) [Size=~1.5GB]. Drop an email to request access.

## Benefits of VAE for Fundus Image Reconstruction
Using a VAE for fundus image reconstruction offers several benefits:
1. **Dimensionality Reduction**: The VAE learns a compact representation of the fundus images, reducing the dimensionality of the data.
2. **Generative Capabilities**: The VAE can generate new fundus images by sampling from the learned latent space.
3. **Image Enhancement**: The VAE can be used to enhance fundus images by reconstructing them with improved clarity and quality.

## Conclusion
In conclusion, using a Variational Autoencoder for fundus image reconstruction is a promising approach to enhance the quality of fundus images. By learning a compact representation of the images, the VAE can reconstruct and enhance fundus images, aiding in the diagnosis and monitoring of eye diseases.
