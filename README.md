# AutomaticImageCaptionGenerator
## A picture is worth a thousand words

## Motivation
The precision of a complex scene description requires a deeper representation of what's actually going on in the scene, the relation among various objects as manifested on the image and the translation of their relationships in one natural-sounding language. Many efforts of establishing such the automatic image camptioning generator combine current state-of-the-art techniques in both *computer vision (CV)* and *natural language processing (NLP)* to form a complete image description approach. We feed one image into this single jointly trained system, and a human readable sequence of words used to describe this image is produced, accordingly. Below shows you how this application can translate from images into words automatically and accurately.

![cnnrnn](https://1.bp.blogspot.com/-O0jjLUCWuhY/VGp6xVUL7uI/AAAAAAAAAcg/wYxwK2AQG4Q/s1600/Screen%2BShot%2B2014-11-17%2Bat%2B2.11.11%2BPM.png)

## Overview
The idea of creating one image caption generator originates from the machine translator among languages. For instance, one recurrent neural network (RNN) transforms a sentence in French into an array of vector representation, and another seperate recurrent neural network takes this vector representation as an input and then transforms it in German. We can mimic this machine translator idea to some extent. For the sake of producing the image caption, the first recurrent neural network used to convert a sentence into a vector is now replaced with one convolutional neural network (CNN) which is in use for *object detection* and *classification*. Generally speaking, the last densely connected layer of one convolutional neural network is fed into the final softmax classifier, assigning a probability that each likely object as displayed in a image. What if we dettach this softmax classifier layer, and we instead regard the last densely connected layer as one rich image embedding layer, being encoded by this pretrained convolutional neural network. This image embedding layer now behaves as the input vector representation, and it's thus fed into one recurrent neural network that is designed to produce phases. Afterward, we can then train one entire single CNN-RNN jointly system directly through a bag of images and associated captions with images. By maximizing the likelihood between ground-truth captions and recurrent neural network predictions, descriptions that recurrent neural network produces to best match each image's captions can be picked up.    

![cnnrnn](https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/encoder_decoder.png)

## Data

The image files splitted into training and validating sets are listed below. The associated caption data with these image files as ground-truth labels is also attached as follow:

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
