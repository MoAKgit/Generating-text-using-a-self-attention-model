
# PyTorch Self-Attention Text Generation.

This repository contains a PyTorch implementation of text generation using the self-attention mechanism. The model is trained on an NVIDIA GeForce RTX 3090 GPU.

## Overview
In this project, we present a simple character-level attention-based decoder for generating reasonably meaningful text.

## Dataset
The dataset for training and validation is provided as a text file in the repository. We divide the dataset into training and validation sets.

To facilitate loading data in batches during training, we have implemented a function called get_batch in utils.py.

## Model
The model architecture is defined in model.py, where we instantiate the GPTLanguageModel().

The overall framework of the model is represented as follows:

![My Image](Overall.PNG) 


The input is a chunk of sequential words coming from the dataset, and the model will predict the next coming words.

The self-attention block emits two vectors for each input, including Queries and Keys.
The query vector represents what the input vector is looking for  
The key vector denotes what the input vector contains.

The input to the model is a chunk of sequential words from the dataset, and the model predicts the next words in the sequence.

The self-attention block in the model emits two vectors for each input: Queries and Keys. The query vector represents what the input vector is looking for, while the key vector denotes what the input vector contains. Self-attention works by giving a high-value output when the query and key are aligned.

The output of the dot product of Query and Key is called the affinity matrix. This matrix is then passed through a softmax function and multiplied by Values. In this project, we use Multi-head attention instead of a single attention mechanism, allowing the model to learn different relationships between words.

## Implementation Details

### Notice 1:
In the case of any deep networks, we probably face optimization issues  (or let's say, vanishing gradient issues). To overcome this problem, in the model, we added some skip connections (it is sometimes called residual pathways ) to help the model with finding better optimal points. This idea was borrowed from the paper "deep residual learning for image recognition".

### Notice 2:
Like other deep learning models, we need to normalize data before and after each attention block. Therefore, in the model, Layer Normalization is applied according to the paper 'Attention Is All You Need'.


### Notice 3:
This project is developed based on character-level input, which is somehow not perfectly suitable for sentence generation. Later, it can be improved by a word-level dictionary, which can possibly boost the performance of the text generating. Also, it can be developed as a machine translation by adding an encoder and also cross-attention in the decoder.

## Acknowledgment

This implementation is adapted from the following source:

https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing



