# Machine Learning Project: Predicting if two frames belong to the same video
CS-433, EPFL
Project realized on google colab with Benoit-Muller (https://github.com/Benoit-Muller) and osflo (https://github.com/osflo) under the supervision of Martin Everaert from the Image and Visual Representation Lab, EPFL, Switzerland.  

## Introduction 
In this project, we tackle the problem of predicting whether two frames come from the same video or not. The motivation is to assess how similar two images are, which could find applications, for instance, for story visualization or video generation. To do so, we use a dataset of videos, we define a class to dynamically extract a pair of frames from the same video or from two different ones. Then, we use the CLIP image encoder (vision transformer) to extract meaningful image features. We then implement two classification methods, a cosine similarity based approach and a neural network with two hidden layers. They take as input our pair of features, and they output the classification prediction, i.e. whether the two frames belong to the same video or not.

## Usage
In order to run this project you need to have a folder containing .mp4 videos, a folder to save your trained model and a folder where you can save your tensor features i.e. the features associated with a given frame output by the CLIP image encoder. The parts in the code where you should adapt the folder path are all commented in the code.  

## Dependencies:
- PyTorch
- cv2
- numpy
- matplotlib

## Structure of the repository

The repository is structured as follows:
```
┣CosineSimilarity.ipynb : notebook implementing the cosine similarity classification method 
┣NeuralNetwork.ipynb : notebook implementing the Neural Network with two hidden layers classification method 
┣PairFrames.py : python file containing the functions needed to dynamically build our dataset from youtube videos   
┣Predicting if two frames are part of the same video.pdf : report of our project describing the methodology, the results and the limits 
┣test_set.ipynb : notebook evaluating our perceptron on the test set 
```
