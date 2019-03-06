"""Training file."""
import os
from utils import Tokenizer, DataGenerator, load_model, GenerateText, str2bool, get_texts
import tensorflow as tf
import argparse
from log import logger
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

    sample_batch = next(iter(train_generator))

    logger.info(f"X batch shape: {sample_batch[0].shape}, y batch shape: {sample_batch[1].shape}")
    logger.info(f"Sample batch text: {tokenizer.decode(sample_batch[0][0])}")

    EMBEDDING_DIM = 512
    NUM_WORDS = tokenizer.num_words

    def lstm_model(seq_len=100, batch_size=None, stateful=True):
        """Language model: predict the next word given the current word."""
        source = tf.keras.Input(
            name='seed', shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)

        embedding = tf.keras.layers.Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_DIM)(source)
        lstm_1 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(
            embedding)
        lstm_2 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(
            lstm_1)
        predicted_char = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(NUM_WORDS, activation='softmax'))(lstm_2)
        model = tf.keras.Model(inputs=[source], outputs=[predicted_char])
        model.compile(
            optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'])
        return model

    training_model = lstm_model(seq_len=seq_len, stateful=False)

    training_model.fit_generator(
        train_generator,
        epochs=1,
    )
    file_path = os.path.join(main_dir, 'models',
                             f'{feature_type}_lstm_{lstm_dim}.hdf5')

    training_model.save_weights(file_path)

    BATCH_SIZE = 1
    PREDICT_LEN = 250

    # Keras requires the batch size be specified ahead of time for stateful models.
    # We use a sequence length of 1, as we will be feeding in one character at a
    # time and predicting the next character.
    prediction_model = lstm_model(seq_len=1, batch_size=BATCH_SIZE, stateful=True)
    prediction_model.load_weights(file_path)

    # We seed the model with our initial string, copied BATCH_SIZE times

    rand_idx = np.random.randint(0, len(train_generator))
    seed = test_generator[rand_idx]

    correct = 0
    # First, run the seed forward to prime the state of the model.
    prediction_model.reset_states()
    for i in range(len(seed.shape[1]) - 1):
        next_probits = prediction_model.predict(seed[:, i:i + 1])
        next_probits = next_probits[:, 0, :]
        pred = np.argmax(next_probits, axis=-1)
        y = seed[:, i + 1]

        correct += (y == pred).sum()

    print(f"Accuracy: {100 * correct / (len(seed.shape[1]) * BATCH_SIZE):.2f}%.")

    # Now we can accumulate predictions!
    predictions = [seed[:, -1:]]
    correct = 0
    for i in range(PREDICT_LEN):
        last_word = predictions[-1]
        next_probits = prediction_model.predict(last_word)

        print(next_probits.shape)

        next_probits = next_probits[:, 0, :]
        print(next_probits.shape)

        pred = np.argmax(next_probits, axis=-1)

        # sample from our output distribution
        next_idx = [
            np.random.choice(256, p=next_probits[i])
            for i in range(BATCH_SIZE)
        ]
        predictions.append(np.asarray(next_idx, dtype=np.int32))

    for i in range(BATCH_SIZE):
        print('PREDICTION %d\n\n' % i)
        p = [predictions[j][i] for j in range(PREDICT_LEN)]
        generated = ''.join([chr(c) for c in p])
        print(generated)
        print()
        assert len(generated) == PREDICT_LEN, 'Generated text too short'


    # model = load_model(seq_len,
    #                    tokenizer.num_words,
    #                    with_embedding=True,
    #                    lstm_dim=lstm_dim)
    #
    # print(model.summary())
    #
    # file_path = os.path.join(main_dir, 'models',
    #                          f'{feature_type}_lstm_{lstm_dim}.hdf5')
    #
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
    parser.add_argument('--lstm-dim', default=128, type=int)
    parser.add_argument('--seq-len', default=28, type=int)
    parser.add_argument('--character-level', default=False, type=str2bool)

    args = parser.parse_args()

    main(args.feature_type, args.main_dir, args.seq_len, args.batch_size, args.lstm_dim,
         args.character_level)
