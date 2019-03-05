"""Training file."""
import os
import logging
from utils import Tokenizer, DataGenerator, load_model
import tensorflow as tf
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SELECTED_BOOKS = ['genesis',
                  'exodus',
                  'leviticus',
                  'deuteronomy',
                  '1_samuel',
                  '2_samuel',
                  '1_kings',
                  '2_kings',
                  'esther',
                  'daniel',
                  '1_chronicles',
                  '2_chronicles',
                  'ecclesiastes']


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
        path = os.path.join(main_dir, 'corpora', 'word', f'{book}.txt')

        with open(path, 'rb') as f:
            texts[book] = f.read().decode()

    full_text = ''.join(texts.values())
    tokenizer = Tokenizer(texts.values(), character_level=character_level)

    train_generator = DataGenerator(tokenizer,
                                    full_text,
                                    seq_len=seq_len,
                                    batch_size=batch_size,
                                    with_embedding=True,
                                    train=True)

    test_generator = DataGenerator(tokenizer,
                                   full_text,
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
                             f'{feature_type}_len_{seq_len}_lstm_{lstm_dim}.hdf5')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss',
                                                    save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    callbacks_list = [checkpoint, early_stopping]

    model.fit_generator(
        train_generator,
        validation_data=test_generator,
        callbacks=callbacks_list,
        epochs=256
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', type=str)
    parser.add_argument('--main-dir', default='./', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lstm-dim', default=128, type=int)
    parser.add_argument('--seq-len', default=64, type=int)
    parser.add_argument('--character-level', default=False)

    args = parser.parse_args()

    main(args.feature_type, args.main_dir, args.seq_len, args.batch_size, args.lstm_dim,
         args.character_level)
