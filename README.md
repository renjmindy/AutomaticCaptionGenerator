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

Linux and macOS: Docker container option
```
To install Docker container with all necessary software installed, follow
```
[instructions](https://hub.docker.com/r/zimovnov/coursera-aml-docker) 
```
After that you should see a Jupyter page in your browser.
```

Windows: Anaconda option

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

## Using Automatic Image Caption Generator

To use automatic image caption generator, follow these steps so that we can prepare resources inside Jupyter Notebooks (for local setups only):

* Click **New** -> **Terminal** and execute: git clone [link](https://github.com/hse-aml/intro-to-dl.git) On Windows you might want to install [Git](https://git-scm.com/download/win) You can also download all the resources as zip archive from GitHub page.
* Close the terminal and refresh Jupyter page, you will see **intro-to-dl folder**, go there, all the necessary notebooks are waiting for you.
* First you need to download necessary resources, to do that open `download_resources.ipynb` and run cells for Keras and your week.

Now you can open a notebook

## Using GPU for offline setup (for advanced users)

## Contributing to <project_name>
<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->
To contribute to <project_name>, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

Thanks to the following people who have contributed to this project:

* [@scottydocs](https://github.com/scottydocs) 📖
* [@cainwatson](https://github.com/cainwatson) 🐛
* [@calchuchesta](https://github.com/calchuchesta) 🐛

You might want to consider using something like the [All Contributors](https://github.com/all-contributors/all-contributors) specification and its [emoji key](https://allcontributors.org/docs/en/emoji-key).

## Contact

If you want to contact me you can reach me at <your_email@address.com>.

## License
<!--- If you're not sure which open license to use see https://choosealicense.com/--->

This project uses the following license: [<license_name>](<link>).
