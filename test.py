"""Testing file."""
import os
from log import logger
from utils import Tokenizer, generate_text, lstm_model, str2bool, get_texts, DataGenerator
import argparse
import numpy as np


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

    texts = get_texts(main_dir, feature_type, character_level)

    tokenizer = Tokenizer(texts.values(), character_level=character_level)

    test_generator = DataGenerator(tokenizer,
                                   tokenizer.full_text,
                                   seq_len=seq_len,
                                   batch_size=batch_size,
                                   with_embedding=True,
                                   train=False)

    sample_batch = next(iter(test_generator))

    logger.info(f"X batch shape: {sample_batch[0].shape}, y batch shape: {sample_batch[1].shape}")
    logger.info(f"Sample batch text: {tokenizer.decode(sample_batch[0][0])}")

    file_path = os.path.join(main_dir, 'models',
                             f'{feature_type}_lstm_{lstm_dim}')

    if character_level:
        file_path += '_character_level'

    file_path += '.h5'

    logger.info(f"Loading {file_path}")

    prediction_model = lstm_model(num_words=tokenizer.num_words,
                                  lstm_dim=lstm_dim,
                                  seq_len=1,
                                  batch_size=test_generator.batch_size,
                                  stateful=True)

    prediction_model.load_weights(file_path)

    rand_idx = np.random.randint(1, len(test_generator))
    seed = test_generator[rand_idx][0]

    generate_text(prediction_model, tokenizer, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', default='word_pos', type=str)
    parser.add_argument('--main-dir', default='./', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lstm-dim', default=256, type=int)
    parser.add_argument('--seq-len', default=32, type=int)
    parser.add_argument('--character-level', default=False, type=str2bool)

    args = parser.parse_args()

    main(args.feature_type, args.main_dir, args.seq_len, args.batch_size, args.lstm_dim,
         args.character_level)
