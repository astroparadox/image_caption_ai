import json
import numpy as np
import os
import pickle
from PIL import Image
import nltk
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu , SmoothingFunction



BASE_DIR = 'coco-2017-dataset/coco2017'
WORKING_DIR = 'working'
max_length = 51 #max_length = max(len(caption.split()) for caption in all_captions)

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



def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text



from gtts import gTTS

def text_to_speech(text, language='en'):
    text = text.replace('startseq', '').replace('endseq', '')  # Remove startseq and endseq
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("output.mp3")
    os.system("start output.mp3")


def generate_caption(image_name):
    under_cap = []
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "train2017", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('##################### Actual #####################')
    for caption in captions:
        under_cap.append(caption)
    true_captions = [under_cap]
    print(true_captions)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('##################### Predicted #####################')
    print(y_pred)
    gen_captions = [y_pred]
    print(gen_captions)
    '''
    text = y_pred
    text_to_speech(text)
    '''
    print('##################### F-SCORE #####################')
    # Smooth function
    smooth = SmoothingFunction().method1  # You can try different smoothing methods
    # Tokenization
    gen_captions_tok = [nltk.word_tokenize(caption.lower()) for caption in gen_captions]
    true_captions_tok = [[nltk.word_tokenize(caption.lower()) for caption in captions] for captions in true_captions]
    # Compute the BLEU score using 4-gram matches with smoothing
    bleu_score = corpus_bleu(true_captions_tok, gen_captions_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    # Compute precision and recall
    precision = bleu_score
    recall = bleu_score / len(gen_captions)
    # Compute F-score
    fscore = 2 * ((precision * recall) / (precision + recall))
    print(fscore)

generate_caption("000000001926.jpg")







