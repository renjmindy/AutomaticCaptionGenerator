# AutomaticImageCaptionGenerator

## Motivation

## Overview

![cnnrnn](https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/encoder_decoder.png)

## Data

* train images http://msvocds.blob.core.windows.net/coco2014/train2014.zip
* validation images http://msvocds.blob.core.windows.net/coco2014/val2014.zip
* captions for both train and validation http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip

## Convolutional Neural Network (CNN) : Image Feature Extraction 

We will use pre-trained InceptionV3 model for CNN encoder and extract its last hidden layer as an embedding:

![cnnrnn](https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/inceptionv3.png)

## Recurrent Neural Network (RNN) : Model Training

Since our problem is to generate image captions, RNN text generator should be conditioned on image. The idea is to use image 
features as an initial state for RNN.

![cnnrnn](https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/encoder_decoder_explained.png)

## LSTM Model 

Below is a demonstration of how the RNN decoder works!

![cnnrnn](https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/flatten_help.jpg)
