import os
import pickle
import numpy as np
import glob
import json
import nltk
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, LSTM, add
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
import tensorflow as tf

BASE_DIR = 'coco-2017-dataset/coco2017'
WORKING_DIR = 'working'

max_length = 35 #max_length = max(len(caption.split()) for caption in all_captions)

model = tf.keras.models.load_model(WORKING_DIR + '/model.h5')

# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

# Load tokenizer
with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)


with open(os.path.join(WORKING_DIR, 'captions.txt'), 'r') as f:
    next(f)  # skip the header
    captions_doc = f.readlines()

# Load the mapping from the JSON file
with open(os.path.join(WORKING_DIR, 'mapping.json'), 'r') as f:
    mapping = json.load(f)

# Prepare train and test data
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.99)
train = image_ids[:split]
test = image_ids[split:]


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text


    
print("[INFO] Validating the model...")
actual, predicted = list(), list()
for key in tqdm(test):
    captions = mapping[key]
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    actual.append(actual_captions)
    predicted.append(y_pred)

print("Blue Score 1 : %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("Blue Score 2 : %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
