# IMDB-Sentiment-Analysis
The main goal is to estimate the sentiment of movie reviews from the Internet Movie Database (IMDb) using simple RNN model.

##What is Sentiment Analysis?
Sentiment analysis is a technique for analysing a piece of text to determine the sentiment contained within it. It accomplishes this by combining machine learning and natural language processing (NLP).

# Example
You can use sentiment analysis to analyse consumer comments, for example you can run sentiment analysis algorithms on such text snippets after collecting input through various mediums such as Twitter and Facebook to assess your customers’ attitudes toward your product.

So Movie Review Analysis is type of customer feedback analysis here, we’ll walk through the steps of creating a model that can perform sentiment analysis on a big movie database. The information was gathered from the Internet Movie Database (IMDb).

# Project Structure
data/: Contains the dataset used for training and testing the model.

models/: Contains the saved models.

notebooks/: Contains Jupyter notebooks with exploratory data analysis, model training, and evaluation.

scripts/: Contains Python scripts for preprocessing data, training models, and evaluating models.

results/: Contains the results of the sentiment analysis, including plots and metrics.

README.md: This file, which provides an overview of the project.

# Installation
To get started with this project, clone the repository and install the necessary dependencies:

git clone https://github.com/your-username/IMDB-Sentiment-Analysis.git
cd IMDB-Sentiment-Analysis
pip install -r requirements.txt

# Usage
Data Preprocessing and Model Training
Run the following script to preprocess the data and train the model:

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Load the IMDb dataset
max_features = 10000  # Vocabulary size
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Print the shape of the data
print(f'Training data shape: {x_train.shape}, Training label shape: {y_train.shape}')
print(f'Testing data shape: {x_test.shape}, Testing label shape: {y_test.shape}')

# Inspect a sample review and its label
sample_review = x_train[0]
sample_label = y_train[0]

print(f"Sample review as (Integers): {sample_review}")
print(f"Sample labels: {sample_label}")

# Map the words
word_index = imdb.get_word_index()
rev_word_index = {value: key for key, value in word_index.items()}

decoded_review = ' '.join([rev_word_index.get(i - 3, '?') for i in sample_review])
print(f"Decoded review: {decoded_review}")

# Pad sequences
max_len = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# Train our Simple RNN
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))  # Embedding layer
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create an instance of EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[earlystopping])

# Save the model
model.save('simple_rnn_imdb.h5')


## Prediction
Load the saved model and predict the sentiment of new reviews:

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Map the words
word_index = imdb.get_word_index()
rev_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')
model.summary()

# Helper function to decode reviews
def decode_review(encoded_review):
    return ' '.join([rev_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# User input and prediction
example_review = "This movie was good. The acting was great and the plot was thrilling."
sentiment, score = predict_sentiment(example_review)

print(f'Review: {example_review}')
print(f'Sentiment: {sentiment}')
print(f'Prediction: {score}')


Results
Include a brief summary of your results here. You can add plots, metrics, or example outputs to give an idea of the model's performance.
Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome all contributions!

# Acknowledgements
IMDb for providing the data.
