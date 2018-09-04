# RNN in PyTorch for Tabular Data
## Written for use with AWS Sagemaker in Deployment

This repo contains a simple PyTorch RNN written for deployment with AWS Sagemaker.

### Directories
training_artifacts: contains all model architecture and training regimen scripts required to train a model


### Training Info
Model can be trained on a single instance of ml.m5.large depending on your dataset size. Larger datasets will benefit from GPU Instances. The entry point is train.py located in the artifacts directory. The entry point accepts a variety of parameters but the defaults have been set to the validated model regimen (determined in research).



### Overview
A recurrent neural network can use many neuron types, by default this repo makes use of the gated recurrent unit (GRU) neuron. GRUs tend to give improved performance in minimal data availability situations. The tensor setup can be set at the top of the training script (number of inputs, output expected, batch size etc).
