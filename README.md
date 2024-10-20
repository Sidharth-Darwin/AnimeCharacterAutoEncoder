# Autoencoder Model for Anime Character Images
=============================================

This repository contains a PyTorch implementation of an autoencoder model designed to learn compact representations of anime character images. The model is trained on the AniWho dataset, a collection of anime character images.

## Autoencoder Model
-----------------

The autoencoder model consists of an encoder and a decoder. The encoder maps the input image to a compact representation, while the decoder maps the compact representation back to the original image. The model is designed to learn a compressed representation of the input images, which can be useful for various computer vision tasks such as image compression, image denoising, and image generation.

## Dataset
--------

The AniWho dataset is used to train and test the autoencoder model. The dataset can be downloaded from [here](https://github.com/mgradyn/AniWho/tree/main/Dataset).

## Usage
-----

To use this repository, simply clone it and install the required dependencies. You can then train the autoencoder model using the provided PyTorch code and test it on the AniWho dataset.

## Files
------

* `model.py`: Contains the autoencoder model definition.
* `train.py`: Contains the training code for the autoencoder model.
* `test.py`: Contains the testing code for the autoencoder model.
* `dataset.py`: Contains the `AniWhoImageDataset` class definition, which loads and preprocesses the AniWho dataset images.

## Dependencies
------------

* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Scikit-learn
