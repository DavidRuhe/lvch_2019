"""Utilities used across the project."""
import tensorflow as tf
from sklearn import utils as skutils
from constants import SELECTED_BOOKS
import numpy as np
import math
from collections.abc import Iterable
import os
from log import logger
import argparse
import string


def get_texts(main_dir, feature_type, character_level):
    texts = {}

    for book in SELECTED_BOOKS:
        path = os.path.join(main_dir, 'corpora', f'{feature_type}', f'{book}.txt')

        with open(path, 'rb') as f:

            text = f.read().decode()

            if not character_level:
                text = text.lower()
                text = text.translate(str.maketrans('', '', string.punctuation))

            texts[book] = text

    return texts


def str2bool(v):
    """Stores boolean in argparse."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Tokenizer:
    """from tensorflow.keras.utils import to_categorical, Sequence"""
    def __init__(self, texts: Iterable, character_level: bool = False):
        self.texts = texts
        print(list(texts)[0].split()[:20])
        self.full_text = "".join(texts)

        self.character_level = character_level

        logger.info(f"Full text sample: {self.full_text[:32]}")

        if character_level:
            self.word_to_ix = {word: i for i, word in enumerate(sorted(set(self.full_text)))}
            self.ix_to_word = {i: word for i, word in enumerate(sorted(set(self.full_text)))}

        else:
            self.word_to_ix = {word: i for i, word in enumerate(sorted(set(
                self.full_text.split())))}
            self.ix_to_word = {i: word for i, word in enumerate(sorted(set(
                self.full_text.split())))}

        self.num_words = len(self.word_to_ix)

        for i, k in enumerate(self.word_to_ix):
            print(k, self.word_to_ix[k])
            if i == 10:
                break

        logger.info(f"Length of tokenizer: {self.num_words}")

    def encode(self, texts: Iterable):
        """Encode an array of texts.

        Parameters
        ----------
        texts: Iterable
            List of texts to encode

        Returns
        -------
            Tuple of encoded texts
        """
        if self.character_level:
            return tuple([self.word_to_ix[word] for word in text] for text in texts)

        else:
            return tuple([self.word_to_ix[word] for word in text.split()] for text in texts)

    def decode(self, encoded: Iterable):
        """Decode an array of integers.

        Parameters
        ----------
        encoded: Iterable
            List of integers to encode

        Returns
        -------
            String of decoded text
        """
        if self.character_level:
            return "".join(self.ix_to_word[int(ix)] for ix in encoded)

        else:
            return " ".join(self.ix_to_word[int(ix)] for ix in encoded)


class DataGenerator(tf.keras.utils.Sequence):
    """Keras data-generator for training."""
    def __init__(self, tokenizer: Tokenizer, text: str, seq_len: int, with_embedding: bool = True,
                 batch_size: int = 512, test_split: float = 0.2, train: bool = False):

        self.encoded = tokenizer.encode([text])[0]

        if train:
            logger.info("Setting up training generator...")

        else:
            logger.info("Setting up testing generator...")

        logger.info(f"Length of encoded texts: {len(self.encoded)}")

        self.batch_size = batch_size
        self.num_words = tokenizer.num_words
        self.with_embedding = with_embedding
        self.seq_len = seq_len

        sentences = []

        for i in range(0, len(self.encoded) - self.seq_len - 1):
            seq = self.encoded[i: i + self.seq_len + 1]
            sentences.append(seq)

        sentences = skutils.shuffle(sentences, random_state=42)

        test_idx = int(test_split * len(sentences))

        if train:
            sentences = sentences[test_idx:]

        else:
            sentences = sentences[:test_idx]

        self.sentences = np.array(sentences).astype(int)

        logger.info(f"Number of sequences: {len(self.sentences)}")

        assert not np.isnan(self.sentences).any()

    def __getitem__(self, idx: int):

        i = idx * self.batch_size

        x_batch = self.sentences[i: i + self.batch_size, :-1]
        y_batch = self.sentences[i: i + self.batch_size, 1:]
        y_batch = np.expand_dims(y_batch, -1)

        if not self.with_embedding:
            x_batch = tf.keras.utils.to_categorical(x_batch, num_classes=self.num_words)

        #         y_batch = to_categorical(y_batch, num_classes=self.num_words)

        return x_batch, y_batch

    def on_epoch_end(self):
        """Reshuffles the set at the end of the epoch."""
        self.sentences, skutils.shuffle(self.sentences)

    def __len__(self):

        return math.ceil(len(self.sentences) / self.batch_size)


# def load_model(seq_len: int,
#                num_words: int,
#                with_embedding: bool = True,
#                stateful: bool = False,
#                batch_size: int = None,
#                lstm_dim: int = 128,
#                embedding_dim: int = 300,
#                return_state: bool = False):
#
#     """Loads a tf.keras LSTM model.
#     Parameters
#     ----------
#     seq_len: length of sequences
#     num_words: number of words in tokenizer
#     with_embedding: use an embedding layer
#     stateful: remember the state after forward pass
#     batch_size: batch size
#     lstm_dim: lstm hidden dimension
#     embedding_dim: embedding dimension
#     return_state: return hidden state
#
#     Returns
#     -------
#     tf.keras LSTM model.
#     """
#
#     if with_embedding:
#         batch_in = tf.keras.layers.Input(name='seed', shape=(seq_len,), batch_size=batch_size)
#
#     else:
#         batch_in = tf.keras.layers.Input(name='seed', shape=(seq_len, num_words),
#                                          batch_size=batch_size)
#
#     if with_embedding:
#         embedding = tf.keras.layers.Embedding(input_dim=num_words, output_dim=embedding_dim,
#                                               input_length=seq_len, batch_size=batch_size)(batch_in)
#
#     else:
#         embedding = batch_in
#
#     if return_state:
#         lstm, hf, cf, hb, cb = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
#             units=lstm_dim, return_state=return_state, stateful=stateful, return_sequences=True))(
#             embedding)
#         dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
#             num_words, activation='softmax'))(lstm)
#
#         model = tf.keras.Model([batch_in], [dense, hf, cf, hb, cb])
#
#     else:
#         lstm = tf.keras.layers.Bidirectional(
#             tf.keras.layers.LSTM(units=lstm_dim, return_state=return_state,
#                                  stateful=stateful, return_sequences=True))(embedding)
#
#         dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
#             num_words, activation='softmax'))(lstm)
#
#         model = tf.keras.Model([batch_in], [dense])
#
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['sparse_categorical_accuracy'])
#
#     return model

def lstm_model(num_words, seq_len=100, batch_size=None, stateful=True):
    """Language model: predict the next word given the current word."""
    source = tf.keras.Input(
        name='seed', shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)

    embedding = tf.keras.layers.Embedding(input_dim=num_words, output_dim=512)(source)
    lstm_1 = tf.keras.layers.LSTM(512, stateful=stateful, return_sequences=True)(
        embedding)
    lstm_2 = tf.keras.layers.LSTM(512, stateful=stateful, return_sequences=True)(
        lstm_1)
    predicted_char = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_words, activation='softmax'))(lstm_2)
    model = tf.keras.Model(inputs=[source], outputs=[predicted_char])
    model.compile(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])
    return model


def load_model(seq_len, num_words, batch_size=None, embedding_dim=300, lstm_dim=128, stateful=False,
               with_embedding=True):
    """Language model: predict the next word given the current word."""

    source = tf.keras.Input(
      name='seed', shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)

    embedding = tf.keras.layers.Embedding(input_dim=num_words, output_dim=embedding_dim)(source)
    lstm_1 = tf.keras.layers.LSTM(lstm_dim, stateful=stateful, return_sequences=True)(embedding)
    lstm_2 = tf.keras.layers.LSTM(lstm_dim, stateful=stateful, return_sequences=True)(lstm_1)
    predicted_char = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_words,
                                                                         activation='softmax'))(lstm_2)
    model = tf.keras.Model(inputs=[source], outputs=[predicted_char])
    model.compile(
      optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),
      loss='sparse_categorical_crossentropy',
      metrics=['sparse_categorical_accuracy'])
    return model

def generate_text(model, tokenizer, generator, predict_len=256):


    # We seed the model with our initial string, copied BATCH_SIZE times


    while True:
        rand_idx = np.random.randint(1, len(generator))

        seed = generator[rand_idx][0]
        print(seed.shape)
        print(rand_idx)

        if seed.shape[0] > 0:
            break

    correct = 0
    # First, run the seed forward to prime the state of the model.
    model.reset_states()
    for i in range(seed.shape[1] - 1):
        print(seed[:, i: i + 1].shape)

        next_probits = model.predict(seed[:, i:i + 1])
        print(next_probits)
        print(next_probits.shape)
        next_probits = next_probits[:, 0, :]
        pred = np.argmax(next_probits, axis=-1)
        y = seed[:, i + 1]

        correct += (y == pred).sum()

    print(f"Accuracy: {100 * correct / (seed.shape[1] * generator.batch_size):.2f}%.")

    # Now we can accumulate predictions!
    predictions = [seed[:, -1:]]
    correct = 0
    for i in range(predict_len):
        last_word = predictions[-1]
        next_probits = model.predict(last_word)

        print(next_probits.shape)

        next_probits = next_probits[:, 0, :]
        print(next_probits.shape)

        pred = np.argmax(next_probits, axis=-1)

        # sample from our output distribution
        next_idx = [
            np.random.choice(tokenizer.num_words, p=next_probits[i])
            for i in range(generator.batch_size)
        ]
        predictions.append(np.asarray(next_idx, dtype=np.int32))

    for i in range(generator.batch_size):
        print('PREDICTION %d\n\n' % i)
        p = [predictions[j][i] for j in range(generator.batch_size)]
        generated = tokenizer.decode(p)
        print(generated)
        print()


class GenerateText(tf.keras.callbacks.Callback):
    """Keras callback to generate text."""
    def __init__(self, generator, tokenizer, weights_path, batch_size):

        super(GenerateText, self).__init__()

        self.generator = generator
        self.tokenizer = tokenizer
        self.weights_path = weights_path
        self.batch_size = batch_size

    def on_train_begin(self, *args, **kwargs):
        """Generates text at the beginning of an epoch."""
        test_model = load_model(seq_len=1, num_words=self.tokenizer.num_words, stateful=True,
                                batch_size=self.batch_size)

        if os.path.exists(self.weights_path):
            test_model.load_weights(self.weights_path)
        generate(self.tokenizer, self.generator, test_model)

    def on_epoch_end(self, *args, **kwargs):
        """Generates text at the end of epoch."""

        print('hi')
        print(self.batch_size)
        test_model = load_model(1, self.tokenizer.num_words, stateful=True, batch_size=self.batch_size)
        test_model.load_weights(self.weights_path)
        generate(self.sample_text, self.tokenizer, test_model)

