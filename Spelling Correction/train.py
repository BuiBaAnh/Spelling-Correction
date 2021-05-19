# -*- coding: utf-8 -*-
import os
from os.path import join, dirname
import itertools

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, LSTM, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from helpers import *

data_path = join(dirname(__file__), 'asset', 'train.txt')
with open(data_path, 'r') as f_r:
    lines = f_r.read().split('\n')

print(len(lines))

NGRAM = 5
BATCH_SIZE = 2048
HIDDEN_SIZE = 256


phrases = itertools.chain.from_iterable(extract_phrases(text) for text in lines)
phrases = [p.strip() for p in phrases if len(p.split()) > 1]

list_ngrams = []
for p in tqdm(phrases):
    if not re.match(pattern, p.lower()):
        continue
    for ngr in gen_ngrams(p, NGRAM):
        if len(' '.join(ngr)) < 36:
            list_ngrams.append(' '.join(ngr))
del phrases
list_ngrams = list(set(list_ngrams))


model = Sequential()
model.add(LSTM(HIDDEN_SIZE, input_shape=(MAXLEN, len(alphabet)), return_sequences=True))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(len(alphabet))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()
# model.load_weights('trained_model/model1_0.3106_0.9016.h5')
train_data, valid_data = train_test_split(list_ngrams, test_size=0.2)

train_generator = generate_data(train_data, batch_size=BATCH_SIZE)
validation_generator = generate_data(valid_data, batch_size=BATCH_SIZE)

checkpointer = ModelCheckpoint(filepath=os.path.join('trained_model/model1_{val_loss:.4f}_{val_accuracy:.4f}.h5'),
                               save_best_only=True, verbose=1)
early = EarlyStopping(patience=2, verbose=1)

model.fit_generator(train_generator, steps_per_epoch=len(train_data)//BATCH_SIZE, epochs=5,
                    validation_data=validation_generator, validation_steps=len(valid_data)//BATCH_SIZE,
                    callbacks=[checkpointer, early])
