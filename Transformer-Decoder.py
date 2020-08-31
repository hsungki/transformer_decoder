"""
Code adapted from Transformer model for language understanding example:
https://www.tensorflow.org/tutorials/text/transformer
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import nltk
import matplotlib.pyplot as plt
import numpy as np
import time
import re
import random
nltk.download('punkt')


# Load text data
file_paths = ['The Old Man and the Sea.txt', 'The Sun Also Rises.txt', 'A Farewell to Arms.txt']
text = ""
for f in file_paths:
    text += open('text/'+f, 'r', encoding='utf-8-sig').read().strip()

text = re.sub(r'[_*]', '', text)
text = re.sub(r'\s+', ' ', text)  
print('The set of characters: ', sorted(set(text)))


# Word tokenization
sentences = nltk.tokenize.sent_tokenize(text)  # Split text into sentences.
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(sentences, target_vocab_size=2**13)
tokenized_sentences = [[tokenizer.vocab_size] + tokenizer.encode(s) + [tokenizer.vocab_size+1] for s in sentences]
print('Number of sentences: ', len(tokenized_sentences))


# Plot the distribution of sentence length.
plot_length_distribution = False
if plot_length_distribution:
    fig, axs = plt.subplots()
    axs.hist(list(map(len, tokenized_sentences)), 20)
    plt.show()


# Limit sentence length
def filter_by_max(_list, _max_length):
    return list(filter(lambda x: len(x) <= _max_length, _list))


max_length = 40
tokenized_sentences = filter_by_max(tokenized_sentences, max_length)
data_size = len(tokenized_sentences)


# Shift sentences by one position to create input and output
data_input = [s[:-1] for s in tokenized_sentences]
data_output = [s[1:] for s in tokenized_sentences]


# Split into training and validation datasets
train_size = (data_size * 90) // 100
train_indices = [*range(data_size)]
random.shuffle(train_indices)
train_input = [data_input[i] for i in train_indices[:train_size]]
train_output = [data_output[i] for i in train_indices[:train_size]]
valid_input = [data_input[i] for i in train_indices[train_size:]]
valid_output = [data_output[i] for i in train_indices[train_size:]]


# Convert to TensorFlow dataset
batch_size = 64
buffer_size = train_size
num_epochs = 100

train_input = tf.keras.preprocessing.sequence.pad_sequences(train_input, padding='post')
train_output = tf.keras.preprocessing.sequence.pad_sequences(train_output, padding='post')
train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(buffer_size)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

valid_input = tf.keras.preprocessing.sequence.pad_sequences(valid_input, padding='post')
valid_output = tf.keras.preprocessing.sequence.pad_sequences(valid_output, padding='post')
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_input, valid_output))
valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True)


# Model parameters
embedding_size = 128
vocab_size = tokenizer.vocab_size + 2  # +2 is for <start> and <end> tokens.
feed_forward_size = 512
num_layers = 2
num_heads = 8
dropout_rate = 0.1


# Positional encoding
def positional_encoding(_seq_length, _embedding_size):
    t = np.arange(0, _seq_length)
    t = t[:, np.newaxis]
    d = np.arange(0, _embedding_size) // 2
    omegas = 1 / (10000 ** (2 * d / _embedding_size))
    omegas = omegas[np.newaxis, :]
    radients = t * omegas
    radients[:, 0::2] = np.sin(radients[:, 0::2])
    radients[:, 1::2] = np.cos(radients[:, 1::2])
    _positional_encoding = radients[np.newaxis, ...]
    return tf.cast(_positional_encoding, dtype=tf.float32)


# Padding mask
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


# Look ahead mask
def create_look_ahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # (seq_len, seq_len)


# Single scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension,
    i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_length_q, depth_q)
        k: key shape ==   (..., seq_length_k, depth_k)
        v: value shape == (..., seq_length_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_length_q, seq_length_k). Defaults to None.

    Restrictions:
        depth_q = depth_k
        seq_length_k = seq_length_v

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_length_q, seq_length_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_length_q, seq_length_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_length_q, depth_v)

    return output, attention_weights


# Multi-head attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, _embedding_size, _num_heads):
        super().__init__()
        self.num_heads = _num_heads
        self.embedding_size = _embedding_size

        assert _embedding_size % self.num_heads == 0

        self.depth = _embedding_size // self.num_heads

        self.wq = tf.keras.layers.Dense(_embedding_size)
        self.wk = tf.keras.layers.Dense(_embedding_size)
        self.wv = tf.keras.layers.Dense(_embedding_size)

        self.dense = tf.keras.layers.Dense(_embedding_size)

    def split_heads(self, x, _batch_size):
        """The shape of input x is (batch_size, seq_length, embedding_size).
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_length, depth)
        """
        x = tf.reshape(x, (_batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        _batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_length_q, depth_q)
        k = self.wk(k)  # (batch_size, seq_length_k, depth_k)
        v = self.wv(v)  # (batch_size, seq_length_v, depth_v)

        q = self.split_heads(q, _batch_size)  # (batch_size, num_heads, seq_length_q, depth_q)
        k = self.split_heads(k, _batch_size)  # (batch_size, num_heads, seq_length_k, depth_k)
        v = self.split_heads(v, _batch_size)  # (batch_size, num_heads, seq_length_v, depth_v)

        # scaled_attention.shape == (batch_size, num_heads, seq_length_q, depth_v)
        # attention_weights.shape == (batch_size, num_heads, seq_length_q, seq_length_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_length_q, num_heads, depth_v)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])

        # (batch_size, seq_length_q, embedding_size)
        concat_attention = tf.reshape(scaled_attention,
                                      (_batch_size, -1, self.embedding_size))

        # output.shape == (batch_size, seq_length_q, embedding_size)
        output = self.dense(concat_attention)

        return output, attention_weights


# Point-wise feed forward network
def point_wise_feed_forward_network(_embedding_size, _feed_forward_size):
    return tf.keras.Sequential([
        # Dense.shape == (batch_size, seq_length, feed_forward_size)
        tf.keras.layers.Dense(_feed_forward_size, activation='relu'),
        # Dense.shape == (batch_size, seq_length, embedding_size)
        tf.keras.layers.Dense(_embedding_size)
    ])


# Single decoder layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, _embedding_size, _num_heads, _feed_forward_size, rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(_embedding_size, _num_heads)

        self.feed_forward = point_wise_feed_forward_network(_embedding_size, _feed_forward_size)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, _look_ahead_mask):
        # attn1.shape == (batch_size, seq_length, embedding_size)
        attn1, attn_weights_block1 = self.mha(x, x, x, _look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # feed_forward_output.shape == (batch_size, seq_length, embedding_size)
        feed_forward_output = self.feed_forward(out1)
        feed_forward_output = self.dropout2(feed_forward_output, training=training)
        # out2.shape == (batch_size, seq_length, embedding_size)
        out2 = self.layernorm2(feed_forward_output + out1)

        return out2, attn_weights_block1


# Transformer Decoder Model
class TransformerDecoder(tf.keras.Model):
    def __init__(self, _num_layers, _embedding_size, _num_heads, _feed_forward_size, _vocab_size,
                 maximum_position_encoding, rate=0.1):
        super().__init__()

        self.embedding_size = _embedding_size
        self.num_layers = _num_layers

        self.embedding = tf.keras.layers.Embedding(_vocab_size, _embedding_size)
        self.pos_encoding = positional_encoding(maximum_position_encoding, _embedding_size)

        self.dec_layers = [DecoderLayer(_embedding_size, _num_heads, _feed_forward_size, rate)
                           for _ in range(_num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        self.final_layer = tf.keras.layers.Dense(_vocab_size)

    def call(self, x, training, _look_ahead_mask):
        seq_length = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, seq_length, embedding_size)
        x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        x += self.pos_encoding[:, :seq_length, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](x, training, _look_ahead_mask)
            attention_weights['decoder_layer{}'.format(i + 1)] = block1

        # x.shape == (batch_size, seq_length, embedding_size)
        # block1.shape == (batch_size, num_heads, seq_length, seq_length)

        final_output = self.final_layer(x)  # (batch_size, tar_seq_len, vocab_size)

        return final_output, attention_weights


# Learning rate, optimizer, and loss function
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, _embedding_size, warmup_steps=4000):
        super().__init__()

        self.embedding_size = _embedding_size
        self.embedding_size = tf.cast(self.embedding_size, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.embedding_size) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(embedding_size)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def accuracy(_tar, _predictions_argmax):
    _value = 0
    _count = 0
    for i in range(_tar.shape[0]):
        current = 0
        j = 0
        while current != vocab_size-1:
            _count += 1
            current = _tar[i][j]
            if current == tf.cast(_predictions_argmax[i][j], tf.int32):
                _value += 1
            j += 1
    return _value, _count


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_loss_history = []

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_loss_history = []


# Create model
transformer_decoder = TransformerDecoder(num_layers, embedding_size,
                                         num_heads, feed_forward_size, vocab_size,
                                         maximum_position_encoding=vocab_size, rate=0.1)


def create_masks(x):
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(x)[1])
    dec_target_padding_mask = create_padding_mask(x)
    return tf.maximum(dec_target_padding_mask, look_ahead_mask)


# Create and load checkpoints if available
checkpoint_dir = 'training_td'
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 transformer_decoder=transformer_decoder)
checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                checkpoint_dir,
                                                max_to_keep=3,
                                                checkpoint_name='ckpoint')
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    # checkpoint_manager.latest_checkpoint is equivalent to
    # tf.train.latest_checkpoint(directory)
    print('Latest checkpoint files are successfully restored.')


# Training and validation step
@tf.function()
def train_step(_inp, _tar):
    combined_mask = create_masks(_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer_decoder(_inp,
                                             True,
                                             combined_mask)
        loss = loss_function(_tar, predictions)

    gradients = tape.gradient(loss, transformer_decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer_decoder.trainable_variables))

    train_loss.update_state(loss)


def valid_step(_inp, _tar):
    combined_mask = create_masks(_inp)
    predictions, _ = transformer_decoder(_inp,
                                         False,
                                         combined_mask)
    loss = loss_function(_tar, predictions)

    valid_loss.update_state(loss)


# Model training
for epoch in range(num_epochs):
    start_time = time.time()

    # Training
    train_loss.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Training Loss {:.4f}'.format(
                   epoch + 1, batch, train_loss.result().numpy()))

    train_loss_history.append(train_loss.result().numpy())

    # Validation
    valid_loss.reset_states()

    for (batch, (inp, tar)) in enumerate(valid_dataset):
        valid_step(inp, tar)

    valid_loss_history.append(valid_loss.result().numpy())

    if (epoch + 1) % 5 == 0:
        checkpoint_save_path = checkpoint_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            checkpoint_save_path))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        train_loss.result().numpy()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start_time))


# Plot model loss
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
x_epoch = np.arange(len(train_loss_history)) + 1
axs.plot(x_epoch, train_loss_history, linewidth=2, label='Training')
axs.plot(x_epoch, valid_loss_history, '--', linewidth=2, label='Validation')
axs.set_xlabel('Epoch', size=15)
axs.set_ylabel('Loss', size=15)
axs.legend(fontsize=10)
plt.tight_layout()
plt.show()
plt.savefig('td_loss.png')


# Model accuracy
train_accuracy = {'value': 0, 'count': 0}
for (batch, (inp, tar)) in enumerate(train_dataset):
    combined_mask = create_masks(inp)
    predictions, _ = transformer_decoder(inp,
                                         False,
                                         combined_mask)
    predictions_argmax = tf.math.argmax(predictions, 2)
    train_value, train_count = accuracy(tar, predictions_argmax)
    train_accuracy['value'] += train_value
    train_accuracy['count'] += train_count

print('Training Accuracy {:.4f}'.format(train_accuracy['value']/train_accuracy['count']))

valid_accuracy = {'value': 0, 'count': 0}
for (batch, (inp, tar)) in enumerate(valid_dataset):
    combined_mask = create_masks(inp)
    predictions, _ = transformer_decoder(inp,
                                         False,
                                         combined_mask)
    predictions_argmax = tf.math.argmax(predictions, 2)
    valid_value, valid_count = accuracy(tar, predictions_argmax)
    valid_accuracy['value'] += valid_value
    valid_accuracy['count'] += valid_count

print('Validation Accuracy {:.4f}'.format(valid_accuracy['value']/valid_accuracy['count']))


# Text generation and attention plot
def text_generator(_model, start, temperature=1.0, plot_attention=False):
    encoded_text = [tokenizer.vocab_size] + tokenizer.encode(start)
    encoded_text = tf.expand_dims(encoded_text, 0)  # Expand by adding a batch dimension.
    max_iteration = tokenizer.vocab_size + 2 - len(encoded_text[0])

    for i in range(max_iteration):
        combined_mask = create_masks(encoded_text)
        prediction, attention = _model(encoded_text,
                                       False,
                                       combined_mask)
        prediction = tf.squeeze(prediction, 0)
        prediction = prediction / temperature
        prediction = tf.random.categorical(prediction, num_samples=1)[-1, 0]
        prediction = tf.squeeze(prediction).numpy()
        prediction = tf.expand_dims([prediction], 0)
        encoded_text = tf.concat([encoded_text, prediction], axis=-1)

        if tf.squeeze(prediction) == tokenizer.vocab_size+1 or tf.squeeze(prediction) == 0:
            encoded_text = tf.squeeze(encoded_text)
            if plot_attention:
                plot_attention_weights(attention, encoded_text, 'decoder_layer2')
            return tokenizer.decode([sw for sw in encoded_text if sw < tokenizer.vocab_size])

    encoded_text = tf.squeeze(encoded_text)
    if plot_attention:
        plot_attention_weights(attention, encoded_text, 'decoder_layer2')
    return tokenizer.decode([sw for sw in encoded_text if sw < tokenizer.vocab_size])


def plot_attention_weights(attention, _encoded_text, layer):
    fig = plt.figure(figsize=(15, 7.5))  # Width and Height
    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:, :], cmap='viridis')

        fontdict = {'fontsize': 9}

        ax.set_xticks(range(len(_encoded_text) - 1))
        ax.set_yticks(range(len(_encoded_text) - 1))

        ax.set_xticklabels(['<start>'] + [tokenizer.decode([i]) for i in _encoded_text[1:-1]],
                           fontdict=fontdict, rotation=90)

        eos = ['<end>'] if _encoded_text[-1] == tokenizer.vocab_size+1 else [tokenizer.decode([_encoded_text[-1]])]
        ax.set_yticklabels([tokenizer.decode([i]) for i in _encoded_text[1:-1]] + eos,
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()
    plt.savefig('attention.png')


# Generate a text from the model
print(text_generator(transformer_decoder, 'I ', 0.9, True))
