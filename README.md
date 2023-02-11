# Overview
This is a repository containing works for the second project to graduate from Udacity Nanodegree "Introduction to Machine Learning with TensorFlow" program in 2022. 

This project consists of two parts. In the first part I used Keras to build an image classifier model using a pre-trained model [MobileNet v2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/). The model I built is exported as a compact model file so it can run in another machine. In the second part, I created a python script to read the model I built in part one to classify flower image.

![Part 1. Build model from a pre-trained model](https://github.com/timmytandian/udacity-project2-tensorflow-classifier/blob/main/readme_asset/project_overview_part1.jpg?raw=true)

![Part 2. Python script to classify flower image](https://github.com/timmytandian/udacity-project2-tensorflow-classifier/blob/main/readme_asset/project_overview_part2.jpg?raw=true)

# Motivation
This repository is intended as a portfolio for datascience work. Recruiter/hiring managers can get idea about my skill and working style as well. Below are skills/key concepts that I practiced in this project:
- neural network and convolutional neural network
- fine tune pre-trained model
- image classification
- keras libraries

# About files in this repository
Below is the explanation of files used in this project.
1. **Project_Image_Classifier_Project.ipynb**: this is the main file I used to build the image classifier model (work part 1). The flow/structure of this file was prepared by Udacity team. I contributed to cells marked with "TODO:". Key activities that I did in this notebook:
    - Load dataset
    - Prepare pipeline for training, validation, testing
    - Model building and training
    - Save/export model as Keras Model
    - Write code for to test classification inference
2. **Project_Image_Classifier_Project.html**: the content of this file is the same as `Project_Image_Classifier_Project.ipynb`. If you want to review the code without jupyter notebook, you can use internet browser to view this file.
3. **predict.py**: a python script to have a keras model to classify a flower image (work part 2). This script accepts command line input argument to perform image classification prediction. This script has dependency to prediction_function.py.
4. **prediction_function.py**: contains some functions required to perform image classification prediction (work part 2).
5. **test_images folder**: this folder contains some images to test the predict.py python script. We feed the path of this folder as input argument for predict.py.
6. **gcolab_epoch[training_epoch]_[time_stamp] folder**: this folder contains keras model. `training_epoch` is the number of epoch to train the model. `time_stamp` is the time stamp when I created this model. We feed the path of this folder as input argument for predict.py.
7. **readme_asset folder**: this folder contains images for this readme. This is not part of the Udacity project.
