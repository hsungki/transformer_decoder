# Transformer Decoder for Text Generation

This repository implements a Transformer-Decoder model for text generation. The code is adapted from [Transformer model for language understanding](
https://www.tensorflow.org/tutorials/text/transformer). 

## Structure of a Transformer Decoder 

The original Transformer model for Machine Translation contains both Encoder and Decoder components. The Encoder processes to-be-translated sentences and the Decoder processes translation results. However, we only need the latter for the purpose of text generation. The overall structure of a Transformer-Decoder is nicely illustrated by the following figure:

> Jay Alammar [The Illustrated GPT-2 (Visualizing Transformer Language Models)](http://jalammar.github.io/illustrated-gpt2/)

![alt text](https://github.com/hsungki/transformer_decoder/blob/master/figures/transformer-decoder-intro.png)

The Decoder layer can be stacked on top of each other. The implementation currently uses 2 layers of them. 

## Model Training 

The following figure shows training and validation losses and accuracies of the model:


## Attention

```python
# Generate a text from the model
print(text_generator(transformer_decoder, 'I ', 0.9, True))
```

![alt text](https://github.com/hsungki/transformer_decoder/blob/master/figures/attention.png)




