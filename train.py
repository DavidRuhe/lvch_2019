"""Training file."""
import os
from utils import Tokenizer, DataGenerator, load_model, GenerateText, str2bool
import tensorflow as tf
import argparse
import string

from constants import SELECTED_BOOKS
from log import logger


def main(feature_type: str, main_dir: str, seq_len: int, batch_size: int, lstm_dim: int,
         character_level: bool = False):
    """
    Parameters
    ----------
    feature_type: the name of the feature
    main_dir: base directory
    seq_len: sequence length
    batch_size: batch size
    lstm_dim: lstm hidden dimension
    character_level: whether tokenizer should be on character level.
    """
    texts = {}

    for book in SELECTED_BOOKS:
        path = os.path.join(main_dir, 'corpora', f'{feature_type}', f'{book}.txt')

        with open(path, 'rb') as f:

            text = f.read().decode()

            if not character_level:
                text = text.translate(str.maketrans('', '', string.punctuation))

            texts[book] = text

    tokenizer = Tokenizer(texts.values(), character_level=character_level)

    train_generator = DataGenerator(tokenizer,
                                    tokenizer.full_text,
                                    seq_len=seq_len,
                                    batch_size=batch_size,
                                    with_embedding=True,
                                    train=True)

    test_generator = DataGenerator(tokenizer,
                                   tokenizer.full_text,
                                   seq_len=seq_len,
                                   batch_size=batch_size,
                                   with_embedding=True,
                                   train=False)

    model = load_model(seq_len,
                       tokenizer.num_words,
                       with_embedding=True,
                       lstm_dim=lstm_dim,
                       batch_size=batch_size)

    logger.info(model.summary())

    file_path = os.path.join(main_dir, 'models',
                             f'{feature_type}_lstm_{lstm_dim}.hdf5')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss',
                                                    save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    generate_text = GenerateText(tokenizer.full_text, tokenizer, file_path)
    callbacks_list = [checkpoint, early_stopping, generate_text]

    model.fit_generator(
        train_generator,
        validation_data=test_generator,
        callbacks=callbacks_list,
        epochs=256
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', default='english', type=str)
    parser.add_argument('--main-dir', default='./', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lstm-dim', default=128, type=int)
    parser.add_argument('--seq-len', default=64, type=int)
    parser.add_argument('--character-level', default=str2bool)

    args = parser.parse_args()

    main(args.feature_type, args.main_dir, args.seq_len, args.batch_size, args.lstm_dim,
         args.character_level)
