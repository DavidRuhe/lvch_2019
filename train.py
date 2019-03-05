"""Training file."""
import os
import logging
from utils import Tokenizer, DataGenerator, load_model
import tensorflow as tf

logging.getLevelName('INFO')
MAIN_DIR = './'


BATCH_SIZE = 64
LSTM_DIM = 128
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

SEQ_LEN = 128

texts = {}

for book in SELECTED_BOOKS:
    path = os.path.join(MAIN_DIR, 'corpora', 'word', f'{book}.txt')

    with open(path, 'rb') as f:
        texts[book] = f.read().decode()

tokenizer = Tokenizer(texts.values(), character_level=True)

train_generator = DataGenerator(tokenizer,
                                texts['genesis'],
                                seq_len=SEQ_LEN,
                                batch_size=BATCH_SIZE,
                                with_embedding=True,
                                train=True)

test_generator = DataGenerator(tokenizer,
                               texts['genesis'],
                               seq_len=SEQ_LEN,
                               batch_size=BATCH_SIZE,
                               with_embedding=True,
                               train=False)

model = load_model(SEQ_LEN,
                   tokenizer.num_words,
                   with_embedding=True,
                   lstm_dim=LSTM_DIM,
                   batch_size=BATCH_SIZE)

logging.info(model.summary())

model_name = 'characters'
file_path = os.path.join(MAIN_DIR, 'models', f'{model_name}.hdf5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
callbacks_list = [checkpoint, early_stopping]


model.fit_generator(
    train_generator,
    validation_data=test_generator,
    callbacks=callbacks_list
)
