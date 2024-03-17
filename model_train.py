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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("[INFO] Load InceptionV3 Model")
model = InceptionV3(weights='imagenet')
model = Model(inputs=model.input, outputs=model.layers[-2].output)

features = {}
image_directory = os.path.join(BASE_DIR,'train2017')

data_splitter=90000

# image features extraction
print("[INFO] Extracting features from images...")
for img_path in tqdm(glob.glob(os.path.join(image_directory, '*.jpg'))[:data_splitter]):  #[:100]
    image = load_img(img_path, target_size=(299, 299))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    features[image_id] = feature


pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
print("[INFO] Features Saved!")

# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)
print("[INFO] Loaded features !")

#print(features['000000000486'])    

# Load COCO captions from JSON file
with open(os.path.join(BASE_DIR, 'annotations', 'captions_train2017.json'), 'r') as f:
    coco_data = json.load(f)

# Extract captions for images with features
mapping = {}
for item in tqdm(coco_data['annotations']):
    image_id = str(item['image_id']).zfill(12)
    if image_id in features:  # Check if features exist for this image
        caption = item['caption']
        if image_id not in mapping:
            mapping[image_id] = []
        # Check if the caption is not empty
        if caption.strip():
            mapping[image_id].append(caption)

# Check if mapping contains empty lines
empty_lines_exist = any([len(captions) == 0 for captions in mapping.values()])
if empty_lines_exist:
    print("WARNING: Mapping contains empty lines.")

# Save captions to captions.txt without empty lines
with open(os.path.join(WORKING_DIR, 'captions.txt'), 'w') as f:
    f.write("image,caption\n")
    for image_id, captions in mapping.items():
        for caption in captions:
            f.write(f"{image_id},{caption}\n")

print("[INFO] Captions Saved to captions.txt")

# Clean captions
print("[INFO] Cleaning Caption Mapping...")
for key, captions in mapping.items():
    for i in range(len(captions)):
        caption = captions[i].lower()
        caption = caption.replace('[^a-z]+', ' ')
        caption = 'startseq ' + caption.strip() + ' endseq'
        captions[i] = caption

# Save mapping to a JSON file
output_file = os.path.join(WORKING_DIR, 'mapping.json')
with open(output_file, 'w') as f:
    json.dump(mapping, f)
print("[INFO] Mapping Saved!")


#print(mapping['000000000486'])

# Tokenize captions
all_captions = [caption for captions in mapping.values() for caption in captions]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)
print("[INFO] Tokenizer saved!")

max_length = max(len(caption.split()) for caption in all_captions)
print("[INFO] Max caption length:", max_length)


# Prepare train and test data
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

# create data generator to get data in smaller batchjs
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
                    # store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0


#model def
print("[INFO] Preparing Encoder model with InceptionV3...")
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

print("[INFO] Preparing an RNN Decoder model with LSTM Neural Network...")
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

print("[INFO] Compiling the model...")
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')


# Train the model
epochs = 20
batch_size = 32
steps = len(train) // batch_size

loss_values = []
print("[INFO] Training the model...")
for i in range(epochs):
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    history = model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    loss_values.append(history.history['loss'])

plt.plot(loss_values, linestyle='-', marker='o', color='blue', label='Training Loss')
plt.title('Training Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, epochs, 5))
plt.yticks(fontsize=12)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Training_Loss.png')

model.save(os.path.join(WORKING_DIR, 'model.h5'))
print("[INFO] Model saved!")

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

#validation / evaluation 

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