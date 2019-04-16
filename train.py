"""Training file."""
import os
from utils import Tokenizer, DataGenerator, lstm_model, GenerateText, str2bool, get_texts
import argparse
from log import logger
import tensorflow as tf


def main(feature_type: str, language: str, domain: str, main_dir: str, seq_len: int,
         batch_size: int, test_batch_size: int, lstm_dim: int, character_level: bool = False):
    """
    Parameters
    ----------
    feature_type: the name of the feature
    main_dir: base directory
    language: language of corpus
    seq_len: sequence length
    batch_size: batch size
    test_batch_size: test batch size
    lstm_dim: lstm hidden dimension
    character_level: whether tokenizer should be on character level.
    """

    texts = get_texts(main_dir, language, feature_type, character_level, domain)

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
                                   batch_size=test_batch_size,
                                   with_embedding=True,
                                   train=False)

    sample_batch = next(iter(train_generator))

    logger.info(f"X batch shape: {sample_batch[0].shape}, y batch shape: {sample_batch[1].shape}")
    logger.info(f"Sample batch text: {tokenizer.decode(sample_batch[0][0])}")

    training_model = lstm_model(num_words=tokenizer.num_words,
                                seq_len=seq_len,
                                lstm_dim=lstm_dim,
                                stateful=False)

    file_path = os.path.join(main_dir, 'models',
                             f'{feature_type}_{language}_lstm_{lstm_dim}')

    if domain:
        file_path += '_' + domain

    if character_level:
        file_path += '_character_level'

    file_path += '.h5'

    training_model.save_weights(file_path)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss',
                                                    save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

    generate_text = GenerateText(test_generator, tokenizer, file_path, lstm_dim)
    callbacks_list = [checkpoint, early_stopping, generate_text]

    training_model.fit_generator(
        train_generator,
        validation_data=test_generator,
        callbacks=callbacks_list,
        epochs=256
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', default='phrase_function', type=str)
    parser.add_argument('--domain', default=None, type=str)
    parser.add_argument('--language', default='hebrew', type=str)
    parser.add_argument('--main-dir', default='./', type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--test-batch-size', default=128, type=int)
    parser.add_argument('--lstm-dim', default=256, type=int)
    parser.add_argument('--seq-len', default=32, type=int)
    parser.add_argument('--character-level', default=False, type=str2bool)

    args = parser.parse_args()

    main(args.feature_type, args.language, args.domain, args.main_dir, args.seq_len,
         args.batch_size, args.test_batch_size, args.lstm_dim, args.character_level)
