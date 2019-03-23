"""Testing file."""
import os
from log import logger
from utils import Tokenizer, generate_text, lstm_model, str2bool, get_texts, DataGenerator
import argparse
import numpy as np
import pickle


def main(feature_type: str, language: str, main_dir: str, seq_len: int, batch_size: int,
         lstm_dim: int, character_level: bool = False):
    """
    Parameters
    ----------
    feature_type: the name of the feature
    language: language of the text.
    main_dir: base directory
    seq_len: sequence length
    batch_size: batch size
    lstm_dim: lstm hidden dimension
    character_level: whether tokenizer should be on character level.
    """

    texts = get_texts(main_dir, language, feature_type, character_level)

    tokenizer = Tokenizer(texts.values(), character_level=character_level)

    samples = {}

    for book in texts:
        len_text = len(texts[book]) if character_level else len(texts[book].split())
        rand_idx = np.random.randint(0, len_text - seq_len, batch_size)

        if character_level:
            samples[book] = tokenizer.encode([texts[book][i: i + seq_len] for i in rand_idx])

        else:
            split_text = texts[book].split()
            samples[book] = tokenizer.encode(
                [" ".join(split_text[i: i + seq_len]) for i in rand_idx]
            )

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
                             f'{feature_type}_{language}_lstm_{lstm_dim}')

    if character_level:
        file_path += '_character_level'

    file_path += '.h5'

    logger.info(f"Loading {file_path}")

    prediction_model = lstm_model(num_words=tokenizer.num_words,
                                  lstm_dim=lstm_dim,
                                  seq_len=1,
                                  batch_size=batch_size,
                                  stateful=True,
                                  return_state=True)

    prediction_model.load_weights(file_path)

    hiddens = {}
    seeds = {}

    for book in samples:
        seed = np.stack(samples[book])
        hf = generate_text(prediction_model, tokenizer, seed, get_hidden=True)
        hiddens[book] = hf
        seeds[book] = seed

    file_name = f'{feature_type}_{language}_lstm_{lstm_dim}_seq_len_{seq_len}'
    if character_level:
        file_name += '_character-level'
    file_name += '.pkl'

    path_out = os.path.join('data', 'hidden_states', file_name)
    with open(path_out, 'wb') as f:
        pickle.dump(hiddens, f)

    logger.info(f"Succesfully saved hidden dimensions to {path_out}")

    path_out = os.path.join('data', 'seeds', file_name)
    with open(path_out, 'wb') as f:
        pickle.dump(seeds, f)

    logger.info(f"Succesfully saved seeds to {path_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', default='word', type=str)
    parser.add_argument('--language', default='english', type=str)
    parser.add_argument('--main-dir', default='./', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lstm-dim', default=512, type=int)
    parser.add_argument('--seq-len', default=32, type=int)
    parser.add_argument('--character-level', default=False, type=str2bool)

    args = parser.parse_args()

    main(args.feature_type, args.language, args.main_dir, args.seq_len, args.batch_size,
         args.lstm_dim, args.character_level)
