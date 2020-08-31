# Transformer-Decoder for Text Generation

This repository implements a Transformer-Decoder model for text generation. The code is adapted from [Transformer model for language understanding](
https://www.tensorflow.org/tutorials/text/transformer). 

## Structure of a Transformer-Decoder 

The original Transformer model for Machine Translation contains both Encoder and Decoder components. The Encoder processes to-be-translated sentences and the Decoder processes translation results. However, we only need the latter for the purpose of text generation. The overall structure of a Transformer-Decoder is nicely illustrated by the following figure:

> Jay Alammar [The Illustrated GPT-2 (Visualizing Transformer Language Models)](http://jalammar.github.io/illustrated-gpt2/)

![alt text](https://github.com/hsungki/transformer_decoder/blob/master/figures/transformer-decoder-intro.png)

The Decoder layer can be stacked on top of each other. The implementation currently uses 2 layers of them. 

## Model Training 

The following figure shows training and validation losses of the model:
![alt text](https://github.com/hsungki/transformer_decoder/blob/master/figures/td_loss.png)

As seen above, we can make several changes to improve its performance in the future. Since there are both high bias and high variance, we can 1) add more Decoder layers and 2) increase the data size for training. 

Nonetheless, the model produces reasonable prediction accuracies. The training and validation accuracies are  

## Text Generation

We can sample a sentence from the model as the following:
```python
# Generate a text from the model
print(text_generator(transformer_decoder, 'I ', 0.9, True))
```
We specify that the generated text will begin with ```'I '```. ```0.9``` refers to the randomnness (or temperature) when selecting a next word during the text generation. The larger the number indicates more randomness. The last input parameter ```True``` indicates whether to generate a heatmap for the last Decoder layer.  

One interesting sample from the model is
```I went out to the kitchen.```
Our model training is based on Ernest Hemingway's The Old Man and the Sea, The Sun Also Rises, and A Farewell to Arms, and it sounds like something that Ernest would write. 

The following is the corresponding heatmap:
![alt text](https://github.com/hsungki/transformer_decoder/blob/master/figures/attention.png)




