"""Utilities used across the project."""
import tensorflow as tf
from sklearn import utils as skutils
import pandas as pd
from constants import SELECTED_BOOKS
import numpy as np
import math
from collections.abc import Iterable
import os
from log import logger
import argparse
import string


def get_texts(main_dir: str, language: str, feature_type: str, character_level: bool,
              domain: str):
    """Load texts from corpus folder.
    Parameters
    ----------
    main_dir: main_dir path
    language: language of corpus
    feature_type: which corpus to read.
    character_level: whether the model is on character level.
    domain: str
        which domain to use: [N, D, Q[

    Returns
    -------
    type: dictionary
        dictionary with name: text.
    """

    if language == 'hebrew':
        path = os.path.join(main_dir, 'corpora', 'main_corpus.csv')

    elif language == 'english':
        path = os.path.join(main_dir, 'corpora', 'english_corpus.csv')

    else:
        raise ValueError(f"{language} is an invalid language choice.")

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError("Run extract_corpora.py or extract_english.py first!")

    df = df.dropna(subset=[feature_type])

    df['book'] = df['book'].str.replace('1_', '')
    df['book'] = df['book'].str.replace('2_', '')
    df.loc[df['book'] == 'ezra', 'book'] = 'ezra-nehemiah'
    df.loc[df['book'] == 'nehemiah', 'book'] = 'ezra-nehemiah'
    df = df[df['book'].isin(SELECTED_BOOKS)]

    df = df[df[feature_type] != df[feature_type].shift()]

    if language == 'hebrew':
        df = df[df['language'] == 'Hebrew']

        if domain:
            assert domain in ['N', 'D', 'Q', '?']
            df = df[df['domain'] == domain]

    if 'language' in df.columns:
        assert df.language.nunique() == 1

    texts = dict(df.groupby('book')[feature_type].apply(list))

    print(len(texts['ecclesiastes']))

    for book in texts:
        text = ' '.join(texts[book])

        if not character_level:
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))

        texts[book] = text

        print(book, len(texts[book].split()))

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

        logger.info(f"Length of tokenizer: {self.num_words}")

    def encode(self, texts: Iterable, errors='raise'):
        """Encode an array of texts.

        Parameters
        ----------
        texts: Iterable
            List of texts to encode
        errors: str
            What to do when encounting errors.

        Returns
        -------
            Tuple of encoded texts
        """

        assert (isinstance(text, str) for text in texts)

        if errors == 'raise':
            if self.character_level:
                return list([self.word_to_ix[word] for word in text] for text in texts)

            else:
                return list([self.word_to_ix[word] for word in text.split(' ')] for text in texts)

        elif errors == 'ignore':
            encoded = []
            for text in texts:

                text_encoded = []
                if self.character_level:
                    for word in text:
                        try:
                            text_encoded.append(self.word_to_ix[word])
                        except KeyError:
                            continue

                else:
                    for word in text.split(' '):
                        try:
                            text_encoded.append(self.word_to_ix[word])
                        except KeyError:
                            continue

                encoded.append(text_encoded)

            return encoded

        else:
            raise ValueError("Provide valid error handling.")

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


def lstm_model(num_words: int,
               seq_len: int = 64,
               batch_size=None,
               stateful: bool = False,
               embedding_dim: int = 300,
               lstm_dim: int = 512,
               return_state: bool = False,
               flatten: bool = False):
    """Language model: predict the next word given the current word.
    Parameters
    ----------
    num_words: num words in corpus
    seq_len: sequence length
    batch_size: batch size
    stateful: whether to retain states after prediction
    embedding_dim: dimension of embedding layer.
    lstm_dim: dimension of lstm layer
    return_state: whether to return the hidden state.

    Returns
    -------
    type: tf.keras.Model
        The language model
    """
    source = tf.keras.Input(
        name='seed', shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)

    embedding = tf.keras.layers.Embedding(input_dim=num_words, output_dim=embedding_dim)(source)

    lstm, hf, cf = tf.keras.layers.LSTM(lstm_dim,
                                        stateful=stateful,
                                        return_state=True,
                                        return_sequences=True)(embedding)

    predicted_char = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_words, activation='softmax'))(lstm)

    if flatten:
        predicted_char = tf.keras.layers.Flatten()(predicted_char)
    if return_state:
        model = tf.keras.Model(inputs=[source], outputs=[predicted_char, hf, cf])
    else:
        model = tf.keras.Model(inputs=[source], outputs=[predicted_char])

    model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=.01),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])
    return model


def generate_text(model, tokenizer, seed, predict_len=256, get_hidden=False, random=False):
    """Generating text with the language model.

    Parameters
    ----------
    model: language model
    tokenizer: tokenizer
    seed: numpy array with seed sequences to prime the model
    predict_len: length of text to predict
    get_hidden: whether to return hidden states.
    random: whether to sample from the probits
    Returns
    -------

    """

    assert seed.shape[0] > 0

    hf = None
    correct = 0
    # First, run the seed forward to prime the state of the model.
    model.reset_states()
    for i in range(seed.shape[1] - 1):
        if not get_hidden:
            next_probits = model.predict(seed[:, i:i + 1])
        else:
            next_probits, hf, cf = model.predict(seed[:, i:i + 1])

        next_probits = next_probits[:, 0, :]
        pred = np.argmax(next_probits, axis=-1)
        y = seed[:, i + 1]
        correct += (y == pred).sum()

    print(f"Accuracy: {100 * correct / (seed.shape[1] * len(seed)):.2f}%.")

    if get_hidden:
        assert hf is not None
        return hf, pred

    predictions = [seed[:, -1:]]
    for i in range(predict_len):
        last_word = predictions[-1]
        next_probits = model.predict(last_word)

        next_probits = next_probits[:, 0, :]

        if random:
            next_idx = [
                np.random.choice(tokenizer.num_words, p=next_probits[i])
                for i in range(len(seed))
            ]
        else:
            next_idx = [np.argmax(next_probits[i]) for i in range(len(seed))]
        predictions.append(np.asarray(next_idx, dtype=np.int32))

    for i in range(len(seed)):
        print(f'\nGenerated text {i + 1}:\n')
        p = [predictions[j][i] for j in range(predict_len)]
        generated = tokenizer.decode(p)
        generated = generated.replace('eos', '\n')
        print(generated)



class GenerateText(tf.keras.callbacks.Callback):
    """Keras callback to generate text."""
    def __init__(self, generator, tokenizer, weights_path, lstm_dim, num_displayed=2):

        super(GenerateText, self).__init__()

        self.generator = generator
        self.tokenizer = tokenizer
        self.weights_path = weights_path
        self.batch_size = generator.batch_size
        self.lstm_dim = lstm_dim
        self.num_displayed = num_displayed

    def on_train_begin(self, *args, **kwargs):
        """Generates text at the beginning of an epoch."""
        test_model = lstm_model(seq_len=1, num_words=self.tokenizer.num_words, stateful=True,
                                batch_size=self.num_displayed, lstm_dim=self.lstm_dim)

        if os.path.exists(self.weights_path):
            test_model.load_weights(self.weights_path)

        rand_idx = np.random.randint(1, len(self.generator))
        seed = self.generator[rand_idx][0][:self.num_displayed]

        generate_text(test_model, self.tokenizer, seed)

    def on_epoch_end(self, *args, **kwargs):
        """Generates text at the end of epoch."""

        test_model = lstm_model(seq_len=1, num_words=self.tokenizer.num_words, stateful=True,
                                batch_size=self.num_displayed, lstm_dim=self.lstm_dim)

        test_model.load_weights(self.weights_path)

        rand_idx = np.random.randint(1, len(self.generator))
        seed = self.generator[rand_idx][0][:self.num_displayed]

        generate_text(test_model, self.tokenizer, seed)

