import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM
# download file
filepath = tf.keras.utils.get_file('shakespeare.txt',
'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# open file, decode it, convert all letters to lower case
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
# we need to convert caracters into numbers and then numbers back into characters
characters = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))
# set of unique characters
# split into sequences
SEQ_LENGTH = 40
STEP_SIZE = 3
# base sentance will be 40 characters long, jum 3 characters from start of sentance to another
sentences = []
next_char = []
# fill up the lists with features and targets of data
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])
# loop can iterate over our text with given length and step size
# Add the initial 40 characters, shift the start by 3, save the next 40.... so on
# needs to be converted into numerical values and the NumPy arrays
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)
# array full of Zeros or Flase VALUES
# now fill arrays
for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1

# time to build a neural network, use optimizer of type RMSprop
model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
# 128 neurons, input shaoe is sentence length and amount of possible characters
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=4)
# learning rate of 0.01, batch size is how many sentances we show model at once
# the model is now trained
# next we will copy a function from the official Keras tutorial
# the higher the temperature the more experimental
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
# will choose the next character, temperature determines riskiness of experiment
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1
        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

print(generate_text(150, 0.2))
print('')
print(generate_text(150, 0.8))
