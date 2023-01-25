# Superresolution GAN

This repository contains the code for a Generative Adversarial Network trained for the task of image super-resolution. The model was trained on CELEBa hq 128x128 and CELEBA HQ 256x256 datasets.

## Demo

A demo of the trained model is available [here](https://superresolution.streamlit.app).



## References

The architecture is based on the following paper:
[Photo-realistic single image super-resolution using a generative adversarial network](https://arxiv.org/abs/1609.04802).
Our generator has a different number of residual blocks, as weel as only one pixelshuffle. As a result, 2x upscaling is performed, instead of 4x as in the paper. We also use a different loss function.
