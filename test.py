"""Training file."""
import os
from log import logger
from utils import Tokenizer, generate, load_model, str2bool
import tensorflow as tf
import argparse
import numpy as np

from constants import SELECTED_BOOKS


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
            texts[book] = f.read().decode()

    full_text = ''.join(texts.values())
    tokenizer = Tokenizer(texts.values(), character_level=character_level)

    model = load_model(seq_len,
                       tokenizer.num_words,
                       with_embedding=True,
                       lstm_dim=lstm_dim,
                       batch_size=batch_size,
                       stateful=True)

    print(model.summary())

    file_path = os.path.join(main_dir, 'models',
                             f'{feature_type}_lstm_{lstm_dim}.hdf5')

    logger.info(f"Loading {file_path}")

    model.load_weights(file_path)

    generate(full_text, tokenizer, model, number_of_seeds=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', default='word', type=str)
    parser.add_argument('--main-dir', default='./', type=str)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--lstm-dim', default=128, type=int)
    parser.add_argument('--seq-len', default=1, type=int)
    parser.add_argument('--character-level', default=False, type=str2bool)

    args = parser.parse_args()

    main(args.feature_type, args.main_dir, args.seq_len, args.batch_size, args.lstm_dim,
         args.character_level)
