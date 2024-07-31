# -*- coding: utf-8 -*-
"""IMDB Review.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RQEsqzBcR3xJvXlCoNkMpSG7ro5dM58B
"""

!pip install tensorflow

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

## load the imbd dataset
max_features= 10000 ## vocabulary size
(x_train, y_train),(x_test, y_test)= imdb.load_data(num_words=max_features)

#print the shape of the data
print(f'Training data shape: {x_train.shape}, Training label shape: {y_train.shape} ')
print(f'Testing data shape: {x_test.shape}, Testing label shape: {y_test.shape} ')

## Inspect a sample review and its label
x_train[0], y_train[0]

sample_review= x_train[0]
sample_label= y_train[0]

print(f"Sample review as (Integers): {sample_review}")
print(f"Sample labels :{sample_label}")

### lets Mapping of the words
word_index= imdb.get_word_index()
rev_word_index= {value:key for key, value in word_index.items()}
rev_word_index

decoded_review = ' '.join([rev_word_index.get(i - 3,'?') for i in sample_review])
decoded_review

max_len= 500
x_train= sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
x_train

x_train[0]

## train our Simple RNN
model = Sequential()
model.add(Embedding(max_features,128, input_length=max_len)) ## Embedding layer
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

## Create an instance of Earlystopping
from tensorflow.keras.callbacks import EarlyStopping
earlystopping=EarlyStopping(monitor='val_loss',patience=5, restore_best_weights=True)
earlystopping

# Train the model with early stopping
history=model.fit(x_train,y_train,epochs=20,batch_size=32,validation_split=0.2,callbacks=[earlystopping])

## save my model file
model.save('simple_rnn_imdb.h5')



import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

### lets Mapping of the words
word_index= imdb.get_word_index()
rev_word_index= {value:key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')
model.summary()

# step 2: Helper function
#function to decode reviews
def decode_review(encoded_review):
    return ' '.join([rev_word_index.get(i - 3,'?') for i in encoded_review])


#Function to represent user input
def preprocess_text(text):
    word = text.lower().split()
    encoded_review = [word_index.get( word, 2) + 3 for word in word]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Prediction Function

def predict_sentiment(review):
    preprocessed_input= preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment= 'Positive' if prediction[0][0]>0.5 else 'Negative'

    return sentiment, prediction[0][0]

# Step 4: User input and Prediction
example_review= "This movie was good The acting was great and plot was thrilling"

sentiment, score = predict_sentiment(example_review)

print(f'Review:{example_review}')
print(f'Sentiment:{sentiment}')
print(f'Prediction:{score}')


