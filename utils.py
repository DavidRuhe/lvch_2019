"""Utilities used across the project."""
import tensorflow as tf
from sklearn import utils as skutils
import numpy as np
import math
from collections.abc import Iterable


class Tokenizer:
    """from tensorflow.keras.utils import to_categorical, Sequence"""
    def __init__(self, texts: Iterable, character_level: bool = False):
        self.texts = texts
        self.full_text = "".join(texts)
        self.character_level = character_level

        if character_level:
            self.word_to_ix = {word: i for i, word in enumerate(set(self.full_text))}
            self.ix_to_word = {i: word for i, word in enumerate(set(self.full_text))}

        else:
            self.word_to_ix = {word: i for i, word in enumerate(set(self.full_text.split()))}
            self.ix_to_word = {i: word for i, word in enumerate(set(self.full_text.split()))}

        self.num_words = len(self.word_to_ix) + 1

        print(f"Length of tokenizer: {self.num_words}")

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


class DataGenerator(tf.keras.utils.Sequence):
    """Keras data-generator for training."""
    def __init__(self, tokenizer: Tokenizer, text: str, seq_len: int, with_embedding: bool = True,
                 batch_size: int = 512, test_split: float = 0.2, train: bool = False):

        self.encoded = tokenizer.encode([text])[0]

        if train:
            print("Setting up training generator...")

        else:
            print("Setting up testing generator...")

        print(f"Length of encoded texts: {len(self.encoded)}")

        self.batch_size = batch_size
        self.num_words = tokenizer.num_words
        self.with_embedding = with_embedding
        self.seq_len = seq_len

        sentences = []
        next_words = []

        for i in range(0, len(self.encoded) - self.seq_len - 1):
            seq = self.encoded[i: i + self.seq_len]
            nw = self.encoded[i + 1: i + self.seq_len + 1]
            sentences.append(seq)
            next_words.append(nw)

        sentences, next_words = skutils.shuffle(sentences, next_words, random_state=42)

        test_idx = int(test_split * len(sentences))

        if train:
            sentences, next_words = sentences[test_idx:], next_words[test_idx:]

        else:
            sentences, next_words = sentences[:test_idx], next_words[:test_idx]

        self.sentences = np.array(sentences).astype(int)
        self.next_words = np.array(next_words).astype(int)

        print(f"Number of sequences: {len(self.sentences)}")

        assert not np.isnan(self.sentences).any()
        assert not np.isnan(self.next_words).any()

    def __getitem__(self, idx: int):

        i = idx * self.batch_size

        x_batch = self.sentences[i: i + self.batch_size]
        y_batch = self.sentences[i: i + self.batch_size]
        y_batch = y_batch[:, :, np.newaxis]

        if not self.with_embedding:
            x_batch = tf.keras.utils.to_categorical(x_batch, num_classes=self.num_words)

        #         y_batch = to_categorical(y_batch, num_classes=self.num_words)
        assert not np.isnan(x_batch).any()
        assert not np.isnan(y_batch).any()

        return x_batch, y_batch

    def on_epoch_end(self):
        """Reshuffles the set at the end of the epoch."""
        self.sentences, self.next_words = skutils.shuffle(self.sentences, self.next_words)

    def __len__(self):

        return math.ceil(len(self.sentences) / self.batch_size)


def load_model(seq_len: int,
               num_words: int,
               with_embedding: bool = True,
               stateful: bool = False,
               batch_size: int = 512,
               lstm_dim: int = 128,
               embedding_dim: int = 300,
               return_state: bool = False):

    """Loads a tf.keras LSTM model.
    Parameters
    ----------
    seq_len: length of sequences
    num_words: number of words in tokenizer
    with_embedding: use an embedding layer
    stateful: remember the state after forward pass
    batch_size: batch size
    lstm_dim: lstm hidden dimension
    embedding_dim: embedding dimension
    return_state: return hidden state

    Returns
    -------
    tf.keras LSTM model.
    """

    if with_embedding:
        batch_in = tf.keras.layers.Input(name='seed', shape=(seq_len,))

    else:
        batch_in = tf.keras.layers.Input(name='seed', shape=(seq_len, num_words),
                                         batch_size=batch_size)

    if with_embedding:
        embedding = tf.keras.layers.Embedding(input_dim=num_words, output_dim=embedding_dim,
                                              input_length=seq_len, batch_size=batch_size)(batch_in)

    else:
        embedding = batch_in

    if return_state:
        lstm, hf, cf, hb, cb = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=lstm_dim, return_state=return_state, stateful=stateful, return_sequences=True))(
            embedding)
        dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
            num_words, activation='softmax'))(lstm)

        model = tf.keras.Model([batch_in], [dense, hf, cf, hb, cb])

    else:
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=lstm_dim, return_state=return_state,
                                 stateful=stateful, return_sequences=True))(embedding)

        dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
            num_words, activation='softmax'))(lstm)

        model = tf.keras.Model([batch_in], [dense])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

    return model
