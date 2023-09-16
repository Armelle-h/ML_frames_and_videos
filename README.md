# Machine Learning Project: Predicting if two frames belong to the same video
CS-433, EPFL
Project realized on google colab with Benoit-Muller (https://github.com/Benoit-Muller) and osflo (https://github.com/osflo)

# Introduction 
In this project, we tackle the problem of predicting whether two frames come from the same video or not. The motivation is to assess how similar two images are, which could find applications, for instance, for story visualization or video generation. To do so, we use a dataset of videos, we define a class to dynamically extract a pair of frames from the same video or from two different ones. Then, we use the CLIP image encoder (vision transformer) to extract meaningful image features. We then implement two classification methods, a cosine similarity based approach and a neural network with two hidden layers. They take as input our pair of features, and they output the classification prediction, i.e. whether the two frames belong to the same video or not.

# Structure of the repository

The repository is structured as follows:
```
┣figures/ : folder containing the figures from algorithm 1, algorithm 2 and performances comparison
┣graphs_algo1.ipynb : notebook implementing algorithm 1 and creating the plots 
┣graphs_algo2.ipynb : notebook implementing algorithm 2 and creating the plots 
┣helpers.py : python file containing helper functions
┣matrices.py : python file containing functions that outputs SPD matrices
┗running_time.ipynb : runs different implementations to compute the trace of a matrix and plots the running times to compare them.
```
