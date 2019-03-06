"""Training file."""
import os
from utils import Tokenizer, DataGenerator, lstm_model, GenerateText, str2bool, get_texts
import argparse
from log import logger


def main(feature_type: str, main_dir: str, seq_len: int, batch_size: int, test_batch_size: int,
         lstm_dim: int, character_level: bool = False):
    """
    Parameters
    ----------
    feature_type: the name of the feature
    main_dir: base directory
    seq_len: sequence length
    batch_size: batch size
    test_batch_size: test batch size
    lstm_dim: lstm hidden dimension
    character_level: whether tokenizer should be on character level.
    """
    texts = get_texts(main_dir, feature_type, character_level)

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

    # train_generator.sentences = train_generator.sentences[:20000]

    sample_batch = next(iter(train_generator))

    logger.info(f"X batch shape: {sample_batch[0].shape}, y batch shape: {sample_batch[1].shape}")
    logger.info(f"Sample batch text: {tokenizer.decode(sample_batch[0][0])}")

    training_model = lstm_model(num_words=tokenizer.num_words,
                                seq_len=seq_len,
                                stateful=False)

    training_model.fit_generator(
        train_generator,
        epochs=1,
    )

    file_path = os.path.join(main_dir, 'models',
                             f'{feature_type}_lstm_{lstm_dim}.h5')

    training_model.save_weights(file_path)

    # checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss',
    #                                                 save_best_only=True)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    #
    # generate_text = GenerateText(test_generator, tokenizer, file_path, batch_size=batch_size)
    # callbacks_list = [checkpoint, early_stopping, generate_text]
    #
    # model.fit_generator(
    #     train_generator,
    #     validation_data=test_generator,
    #     callbacks=callbacks_list,
    #     epochs=256
    # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', default='english', type=str)
    parser.add_argument('--main-dir', default='./', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--test-batch-size', default=8, type=int)
    parser.add_argument('--lstm-dim', default=128, type=int)
    parser.add_argument('--seq-len', default=28, type=int)
    parser.add_argument('--character-level', default=False, type=str2bool)

    args = parser.parse_args([])

    main(args.feature_type, args.main_dir, args.seq_len, args.batch_size,
         args.test_batch_size, args.lstm_dim, args.character_level)
