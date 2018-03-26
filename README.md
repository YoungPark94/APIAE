# `APIAE`: Adpative Path Integral Autoencoder

This is a tensorlfow implementation for the demonstration of Pendulum experiment of the paper: "Adaptive Path-Integral Approach for Representation Learning and Planning of State Space Models"

https://openreview.net/pdf?id=HyoOUrkwz

## Requirements

- Python 3
- Tensorflow 3
- Numpy
- Pickle

## Instructions

1. To train APIAE for pendulum example, run 'train_pendulum.py'.
2. To run pendulum planning demo by using trained APIAE, run 'demo_pendulum.py'.
Without any modification, this code load the hyperparatmers (i.e. weights of neural networks) from 'weights_demo.pkl'.
But you may change the name of file if you want to test with new APIAE network trained from step 1.
