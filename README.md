# Automatic Image Caption Generator
## A picture is worth a thousand words

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here 
![GitHub repo size](https://img.shields.io/github/repo-size/scottydocs/README-template.md)
![GitHub contributors](https://img.shields.io/github/contributors/scottydocs/README-template.md)
![GitHub stars](https://img.shields.io/github/stars/scottydocs/README-template.md?style=social)
![GitHub forks](https://img.shields.io/github/forks/scottydocs/README-template.md?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/scottydocs?style=social) --->

## Motivation

Automatic image caption generator is a widely used deep learning application that combines `convolutional neural network` and `recurrent neural network` to allow `computer vision` tasks being `described in written statement` and/or in `sound of text`.

The precision of a complex scene description requires a deeper representation of what's actually going on in the scene, the relation among various objects as manifested on the image and the translation of their relationships in one natural-sounding language. Many efforts of establishing such the automatic image captioning generator combine current state-of-the-art techniques in both *computer vision (CV)* and *natural language processing (NLP)* to form a complete image description approach. We feed one image into this single jointly trained system, and a human readable sequence of words used to describe this image is produced, accordingly. Below shows you how this application can translate from images into words automatically and accurately.

![cnnrnn](https://1.bp.blogspot.com/-O0jjLUCWuhY/VGp6xVUL7uI/AAAAAAAAAcg/wYxwK2AQG4Q/s1600/Screen%2BShot%2B2014-11-17%2Bat%2B2.11.11%2BPM.png)

## Prerequisites

Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* You have installed `Jupyter Notebooks` for **offline** analysis. There're two options to setup the Jupyter Notebooks locally: Docker container and Anaconda. You will need a computer with at least 4GB of RAM.
* Alternatively, you can run all tasks **online** via `Google Colab`. Google has released its own flavour of Jupyter called Colab, which has free GPUs!
* You have a `Windows/Linux/Mac` machine.

## Installing for Offline Instructions

To install Automatic Image Caption Generator, follow these steps:

### Linux and macOS: Docker container option

```
To install Docker container with all necessary software installed, follow
```

[instructions](https://hub.docker.com/r/zimovnov/coursera-aml-docker) After that you should see a Jupyter page in your browser.

### Windows: Anaconda option

```
We highly recommend to install docker environment, but if it's not an option, you can try to install the necessary python modules with Anaconda.
```

* First, install Anaconda with **Python 3.5+** from [here](https://www.anaconda.com/products/individual).
* Download `conda_requirements.txt` from [here](https://github.com/ZEMUSHKA/coursera-aml-docker/blob/master/conda_requirements.txt).
* Open terminal on Mac/Linux or "Anaconda Prompt" in Start Menu on Windows and run:

```
conda config --append channels conda-forge
conda config --append channels menpo
conda install --yes --file conda_requirements.txt
```

To start Jupyter Notebooks run `jupyter notebook` on Mac/Linux or "Jupyter Notebook" in Start Menu on Windows.

After that you should see a Jupyter page in your browser.

## Using GPU for offline setup (for advanced users)

* If you have a Linux host you can try these [instructions](https://github.com/ZEMUSHKA/coursera-aml-docker#using-gpu-in-your-container-linux-hosts-only) for Docker
* The easiest way is to go with Anaconda setup, that doesn't need virtualization and thus works with a GPU on all platforms (including Windows and Mac). You will still have to install NVIDIA GPU driver, CUDA toolkit and CuDNN (requires registration with NVIDIA) on your host machine in order for [TensorFlow to work with your GPU](https://www.tensorflow.org/install/gpu). It can be hard to follow, so you might choose to stick to a CPU version, which is also fine for the purpose of this course.

## Running on Google Colab

Google has released its own flavour of Jupyter called Colab, which has free GPUs!

Here's how you can use it:

* Open [link](https://colab.research.google.com), click Sign in in the upper right corner, use your Google credentials to sign in.
* Click **GITHUB** tab, paste [link](https://github.com/hse-aml/intro-to-dl) and press Enter
* Choose the [notebook](wee6/week6_final_project_image_captioning_clean.ipynb) you want to open
* Click **File -> Save a copy in Drive**... to save your progress in Google Drive
* Click **Runtime** -> **Change runtime type** and select `GPU` in Hardware accelerator box
* **Execute** the following code in the first cell that downloads dependencies (change for your week number):

```
! shred -u setup_google_colab.py
! wget https://raw.githubusercontent.com/hse-aml/intro-to-dl/master/setup_google_colab.py -O setup_google_colab.py
import setup_google_colab
setup_google_colab.setup_week6()
```

* If you run many notebooks on Colab, they can continue to eat up memory, you can kill them with `! pkill -9 python3` and check with `! nvidia-smi` that GPU memory is freed.

**Known issues:**

* Blinking animation with `IPython.display.clear_output()`. It's usable, but still looking for a workaround.

## Getting started with using automatic image caption generator

* **Usage** to use automatic image caption generator, follow these steps so that we can prepare resources inside Jupyter Notebooks (for local setups only):

  - Click **New** -> **Terminal** and execute: git clone [link](https://github.com/hse-aml/intro-to-dl.git) On Windows you might want to install [Git](https://git-scm.com/download/win) You can also download all the resources as zip archive from GitHub page.
  - Close the terminal and refresh Jupyter page, you will see **intro-to-dl folder**, go there, all the necessary notebooks are waiting for you.
  - First you need to download necessary resources, to do that open `download_resources.ipynb` and run cells for Keras and your week.

Now you can open a notebook by cloning this [repo](https://github.com/renjmindy/AutomaticImageCaptionGenerator) (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

* **[Procedures](https://github.com/renjmindy/AutomaticImageCaptionGenerator/blob/master/capstone_project_image_caption_generator.ipynb)**

  - Set-up environment: run [setup_google_colab.py](https://github.com/renjmindy/AutomaticImageCaptionGenerator/blob/master/setup_google_colab.py)
    + run `setup_common` function where [keras_utils.py](https://github.com/renjmindy/AutomaticImageCaptionGenerator/blob/master/keras_utils.py), [download_utils.py](https://github.com/renjmindy/AutomaticImageCaptionGenerator/blob/master/download_utils.py) and [tqdm_utils.py](https://github.com/renjmindy/AutomaticImageCaptionGenerator/blob/master/tqdm_utils.py) are downloaded via `download_github_code` function
    + run `download_github_code` function to download [utils.py](https://github.com/renjmindy/AutomaticImageCaptionGenerator/blob/master/utils.py) that is in specific use for automatic image caption generator 
    + run `download_utils` file to download image embedding pickles, compressed image and caption zip files   
    + run `setup_keras` function to execute [download_utils](https://github.com/renjmindy/AutomaticImageCaptionGenerator/blob/master/download_utils.py) file where `download_all_keras_resources` function downloads pre-trained inceptionV3 model

  - Prepare data (Extract image features/Extract captions for images) by extracting image and caption samples from compressed files
    + run `download_utils.py` to download [train2014_sample.zip](https://github.com/hse-aml/intro-to-dl/releases/tag/v0.1) and [val2014_sample.zip](https://github.com/hse-aml/intro-to-dl/releases/tag/v0.1)
    + images:
      * write `get_cnn_encoder` function to obtain one selective pre-trained model without the classifier 
      pre-trained [InceptionV3](https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html) model for CNN encoder![captiongen](https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/inceptionv3.png)
      * get training and validation images through [apply_model](https://github.com/renjmindy/AutomaticImageCaptionGenerator/blob/master/utils.py) function where corresponding embedding features to images are extracted
    + captions:
      * write `get_captions_for_fns` funtion to create one dictionary where key stands for each image's file name, and value is a list of corrsponding captions to one specific kay image 
      * write `show_trainig_example` function to look at training example (each has 5 captions)
    
  - image pre-processing and text cleaning (Prepare captions for training)
    RNN should be conditioned on images to generate image captions behaving as RNN initial hidden state![captiongen](https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/encoder_decoder_explained.png)
    + captions:
      * write `split_sentence` function to split one sentence into tokens, i.e. lowercased words
      * write `generate_vocabulary` function to select most frequent tokens that occur 5 times or more from training captions
      * write `caption_tokens_to_indices` function to construct a multi-layer of arrays in which each associated caption with one given photo image is chopped into an array of words. Every image allows up to 5 arrays of semented tokens (words), every of which has `START` and `END` tokens being added from head and tail of one tokenized caption, respectively. 
        ```
        captions argument is an array of arrays: [ [ "image1 caption1", "image1 caption2", ... ], [ "image2 caption1", "image2 caption2", ... ], ... ]
        [
          [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
          ],
          ...
        ]
        ```
        `batch_captions` turns out to be an array of arrays: [ [vocab[START], ..., vocab[END]], [vocab[START], ..., vocab[END]], ... ]
      * write `batch_captions_to_matrix` function to convert `batch_captions` into an equal length of `matrix` for every given image. Since associated captions with one given image might have different lengths, we add `PAD` token to make shorter caption(s) become as long as the longest caption. 
      ```
      Input example: 
                    [[1, 2, 3], [4, 5]]
      Output examples:
                    1. if max_len=None or 100
                       np.array([[1, 2, 3], [4, 5, pad_idx]]) 
                    2. if max_len=2
                       np.array([[1, 2], [4, 5]]) 
       ```
       
  - model training (Training) 
    + write `decoder` class in which we compute cross-entropy between `flat_ground_truth` and `flat_token_logits` predicted by LSTM
      `decoder` class describes how one specific type of RNN architectures, LSTM (Long Short Term Memory), works![captiongen](https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/flatten_help.jpg)
      * use `bottleneck` here to reduce the number of parameters : image embedding -> bottleneck -> LSTM initial state
      * use `bottleneck` here to reduce model complexity : LSTM output -> logits bottleneck -> logits for next token prediction
      * embed all words (word -> embedding) to be used as LSTM input of ground truth tokens : `word_embeds` as context for next token prediction
      * know all inputs for LSTM and can get all the hidden states with one tensorflow operation (tf.nn.dynamic_rnn)
      * calculate `token_logits` for all the hidden states step-by-step as follows:
        (1) calculate logits for next tokens
        (2) predict next tokens for each time step
        (3) need to know where we have real tokens (not `PAD` token)
        (4) not propagate output `PAD` tokens for the loss computation : fill with 1.0 for real, non-`PAD` tokens and 0.0 otherwise
      * compute cross-entropy between `ground_truth` and `token_logits` predicted by LSTM : average `xent` over tokens with nonzero `loss_mask` so that we don't account misclassification of `PAD` tokens which were added simply for the batching purpose
      * define optimizer operation to minimize the loss
      * save/load network weights     
    + write `generate_batch` function to generate a random batch of size `batch_size` via random sampling of images and captions for them
    
  - model application to testing samples (Applying)
    + write `final_model` class which works as follows:
      * take an image as an input and embed it : run `get_cnn_encoder` to pass images through CNN encoder in order to obtain image embedding files
      * condition LSTM on that embedding : run `decoder` class to initialize LSTM state being conditioned on images
      * predict the next token given a START input token : run `decoder` class to get current word embedding being passed to LSTM cell to produce new LSTM states
      * use predicted token as an input at next time step : run `decoder` class to compute logits and probabilities for next token
      * iterate until you predict an END token : `one_step` yields probabilities of next token and meanwhile updates LSTM hidden state
    + write `generate_caption` function to generate caption for given image
    + write `apply_model_to_image_raw_bytes` and `show_valid_example` functions to look at validation prediction example
      
* **Files** This [repository](https://github.com/renjmindy/FaceDetectors/tr) consist of multiple files:

  - `capstone_project_image_caption_generator.ipynb` -- main task, read and work
  - `setup_google_colab.py` -- initialize environmental settings and call python codes as below
  - `utils.py` -- execute image pre-processing and embedding feature extraction to save and read embedding files for future use
  - `keras_utils.py` -- prepare keras functions useful for model training 
  - `download_utils.py` -- download all necessary files, e.g. python codes and compressed data samples
  - `tqdm_utils.py` -- show the completion progress of model-training tasks

* **Dataset**

  - [train images](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)
  - [validation images](http://msvocds.blob.core.windows.net/coco2014/val2014.zip)
  - [train and validation captions](http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip)
  - train sample images: [train2014_sample.zip](https://github.com/hse-aml/intro-to-dl/releases/tag/v0.1)
  - validation sample images: [val2014_sample.zip](https://github.com/hse-aml/intro-to-dl/releases/tag/v0.1)
  
## Contributing to Automatic Image Caption Generator
<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->
To contribute to `automatic image caption generator`, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin AutomaticImageCaptionGenerator/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

### Methods Used

* Multi-layered Perceptions (MLPs)
* Convolution Neural Networks (CNN)
* Recurrent Neural Networks (RNN)
* Transfer Learning
* Gradient Descent
* Backpropagation
* Overfitting
* Probability
* Text Processing

### Technologies

* Traditional Image Processing Techniques, e.g. use of OpenCV, Scikit-image (skimage)
* Python (3.x)
* Pandas, jupyter
* Keras (2.2 or higher)
* TensorFlow (as backend)
* TensorFlow.compat.v1 (disable_v2_behavior)
* Scikit-learn
* Matplotlib
* NumPy

## Project Overview

The idea of creating one image caption generator originates from the machine translator among languages. For instance, one recurrent neural network (RNN) transforms a sentence in French into an array of vector representation, and another separate recurrent neural network takes this vector representation as an input and then transforms it in German. We can mimic this machine translator idea to some extent. For the sake of producing the image caption, the first recurrent neural network used to convert a sentence into a vector is now replaced with one convolutional neural network (CNN) which is in use for *object detection* and *classification*. Generally speaking, the last densely connected layer of one convolutional neural network is fed into the final softmax classifier, assigning a probability that each likely object as displayed in a image. What if we detach this softmax classifier layer, and we instead regard the last densely connected layer as one rich image embedding layer, being encoded by this pretrained convolutional neural network. This image embedding layer now behaves as the input vector representation, and it's thus fed into one recurrent neural network that is designed to produce phases. Afterward, we can then train one entire single CNN-RNN jointly system directly through a bag of images and associated captions with images. By maximizing the likelihood between ground-truth captions and recurrent neural network predictions, descriptions that recurrent neural network produces to best match each image's captions can be picked up.    

![cnnrnn](https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/encoder_decoder.png)

## Needs of this project

* image and caption data pre-processing, along with tokenized caption cleaning
* image feature extraction from pre-trained model  
* LSTM model training to predict tokenized caption with image features as initial hidden state  

## Featured Notebooks/Analysis/Deliverables/Blog Posts

### Blog Posts

* [Why is CNN (if MLP available)?](https://renjmindy.github.io/why_is_cnn_if_mlp_available)
* [What're modern CNN architectures?](https://renjmindy.github.io/introduction_to_convolutional_neural_network_cnn)
* [How is CNN architecture built up from scratch?](https://renjmindy.github.io/introduction_to_convolutional_neural_network_cnn)

## Contact

If you want to contact me you can reach me at <jencmhep@gmail.com>.
