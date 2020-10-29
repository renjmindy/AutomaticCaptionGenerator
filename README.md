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

The precision of a complex scene description requires a deeper representation of what's actually going on in the scene, the relation among various objects as manifested on the image and the translation of their relationships in one natural-sounding language. Many efforts of establishing such the automatic image camptioning generator combine current state-of-the-art techniques in both *computer vision (CV)* and *natural language processing (NLP)* to form a complete image description approach. We feed one image into this single jointly trained system, and a human readable sequence of words used to describe this image is produced, accordingly. Below shows you how this application can translate from images into words automatically and accurately.

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

## Getting Started Using Automatic Image Caption Generator

To use automatic image caption generator, follow these steps so that we can prepare resources inside Jupyter Notebooks (for local setups only):

* Click **New** -> **Terminal** and execute: git clone [link](https://github.com/hse-aml/intro-to-dl.git) On Windows you might want to install [Git](https://git-scm.com/download/win) You can also download all the resources as zip archive from GitHub page.
* Close the terminal and refresh Jupyter page, you will see **intro-to-dl folder**, go there, all the necessary notebooks are waiting for you.
* First you need to download necessary resources, to do that open `download_resources.ipynb` and run cells for Keras and your week.

Now you can open a notebook by cloning this [repo](https://github.com/renjmindy/AutomaticImageCaptionGenerator) (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

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

* Multi-layered Perceptrons (MLPs)
* Convolution Neural Networks (CNN)
* Recurrent Neural Networks (RNN)
* Transfer Learning
* Gradient Descent
* Backpropagation
* Overfitting
* Probability
* Text Processing

### Technologies

* Python (3.x)
* Pandas, jupyter
* Keras (2.2 or higher)
* TensorFlow (as backend)
* Scikit-learn
* Matplotlib
* NumPy

## Project Overview
The idea of creating one image caption generator originates from the machine translator among languages. For instance, one recurrent neural network (RNN) transforms a sentence in French into an array of vector representation, and another seperate recurrent neural network takes this vector representation as an input and then transforms it in German. We can mimic this machine translator idea to some extent. For the sake of producing the image caption, the first recurrent neural network used to convert a sentence into a vector is now replaced with one convolutional neural network (CNN) which is in use for *object detection* and *classification*. Generally speaking, the last densely connected layer of one convolutional neural network is fed into the final softmax classifier, assigning a probability that each likely object as displayed in a image. What if we dettach this softmax classifier layer, and we instead regard the last densely connected layer as one rich image embedding layer, being encoded by this pretrained convolutional neural network. This image embedding layer now behaves as the input vector representation, and it's thus fed into one recurrent neural network that is designed to produce phases. Afterward, we can then train one entire single CNN-RNN jointly system directly through a bag of images and associated captions with images. By maximizing the likelihood between ground-truth captions and recurrent neural network predictions, descriptions that recurrent neural network produces to best match each image's captions can be picked up.    

![cnnrnn](https://github.com/hse-aml/intro-to-dl/blob/master/week6/images/encoder_decoder.png)

## Needs of this project

- frontend developers
- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- writeup/reporting
- etc. (be as specific as possible)

## Featured Notebooks/Analysis/Deliverables/Blog Posts

### Blog Posts
* [Why is CNN (if MLP available)?](https://renjmindy.github.io/why_is_cnn_if_mlp_available)
* [What're modern CNN architectures?](https://renjmindy.github.io/introduction_to_convolutional_neural_network_cnn)
* [How is CNN architecture built up from scratch?](https://renjmindy.github.io/introduction_to_convolutional_neural_network_cnn)
* [Diagnosis of COVID-19 alike Viral Pneumonia:Building CNN from Scratch for Pneumonia Diagnosis by Classifying Chest X-Ray Images in Patients](https://renjmindy.github.io/diagnosis_of_covid-19_alike_viral_pneumonia)

## Contact

If you want to contact me you can reach me at <jencmhep@gmail.com>.
