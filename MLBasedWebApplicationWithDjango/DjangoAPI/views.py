from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import ComplaintForm
from .models import FromDataBase
from random import randint

#imports for preprocessing
import nltk
import string
import re
from nltk.corpus import stopwords
import nltk
import zeyrek
from zemberek import (
    TurkishSpellChecker,
    TurkishSentenceNormalizer,
    TurkishSentenceExtractor,
    TurkishMorphology,
    TurkishTokenizer
)

# imports for tensorflow model
import numpy as np
import pandas as pd
import tensorflow as tf
import fasttext
from transformers import pipeline
import transformers as trfs
import sklearn.model_selection as ms
import sklearn.preprocessing as p
from official.nlp import optimization
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt, mpld3

# imports for Summarization model
import torch
torch.cuda.empty_cache()
from transformers import BertTokenizerFast, EncoderDecoderModel

#PRERPOCESSING--------------------------------

#Download stopwords and punkt list
nltk.download('stopwords')
nltk.download('punkt')


puncts_list = string.punctuation
stopword_list = stopwords.words('turkish')


#Sentence boundary detection and then normalization
morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)
extractor = TurkishSentenceExtractor()

def normalize(text):
    sentences = extractor.from_paragraph(text)
    temp_sentences = ''
    for sentence in sentences:
        temp_sentences += normalizer.normalize(sentence) + ' '
    context = {'normalized': temp_sentences}
    return context


def remove_num_and_stopwords(text):
    text = normalize(text)['normalized']
    text = text.lower()
    temp = ''
    for word in text.split():
        if word not in stopword_list and not word.isnumeric():
            temp += str(word) + ' '
    context = {'removed_stop' : temp}
    return context


def remove_puncts(text):
    text = remove_num_and_stopwords(text)['removed_stop']
    temp = ''
    for char in text:
        if char not in puncts_list:
            temp += char
    temp = temp.lower()
    context = {'removed_puncts' : temp}
    return context

#Morphologic analysis
analyzer = zeyrek.MorphAnalyzer()

def lemmatize(text):
    text = remove_puncts(text)['removed_puncts']
    temp = ''
    for word in text.split():
        temp += analyzer.lemmatize(word)[0][1][0] + ' '
    temp = temp.lower()
    context = {'lemmatized' : temp}
    return context

#BERT MODEL----------------------------------------

# Parameters for Model

# Max length of encoded string(including special tokens such as [CLS] and [SEP]):
MAX_SEQUENCE_LENGTH = 128 
# Standard BERT model with lowercase chars only:
PRETRAINED_MODEL_NAME = 'dbmdz/bert-base-turkish-128k-cased'
# Batch size for fitting:
BATCH_SIZE = 48 
# Number of epochs:
EPOCHS=20
# Learing rate:
LR = 3e-5
# Setting seeds for repeatability
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
EPOCHLEN = 10

#creating tokenizer here to save time
tokenizer = trfs.BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=False)


def encode(X, tokenizer):
    return tokenizer.encode_plus(
    text = X,
    max_length=MAX_SEQUENCE_LENGTH, # set the length of the sequences
    truncation=True,
    padding='max_length',
    add_special_tokens=True, # add [CLS] and [SEP] tokens
    return_attention_mask=True,
    return_token_type_ids=False, # not needed for this type of ML task
    return_tensors='tf'
)


def create_altered_model(max_sequence, model_name, num_labels):
    bert_model = trfs.TFBertForSequenceClassification.from_pretrained(model_name, num_labels)
    
    input_ids = tf.keras.layers.Input(shape=(max_sequence,), dtype=tf.int32, name='input_ids')

    
    attention_mask = tf.keras.layers.Input((max_sequence,), dtype=tf.int32, name='attention_mask')
    
    # Use previous inputs as BERT inputs:
    output = bert_model([input_ids, attention_mask])[0]

    # We can also add dropout as regularization technique:
    output = tf.keras.layers.Dropout(0.4, name='extra_dropout')(output)

    # Provide number of classes to the final layer:
    output = tf.keras.layers.Dense(1, activation=None, name='output')(output)

    # Final model:
    model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
    return model


#create model here to save time. If we create it in the view function,
#it will create model from the beginning every time we refresh the page.
model = create_altered_model(MAX_SEQUENCE_LENGTH, PRETRAINED_MODEL_NAME, 1)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()
steps_per_epoch = EPOCHLEN
num_train_steps = steps_per_epoch * EPOCHS
num_warmup_steps = int(0.1*num_train_steps)
ckpoint = ModelCheckpoint('saved_models/bert-cased-128-043/bert-cased-128-043', 
                          monitor='val_binary_accuracy', 
                          verbose=1, save_best_only=True,
                          save_weights_only=True, mode='max')
init_lr = LR

optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

model.compile(optimizer=optimizer,loss=loss, metrics=metrics)
model.load_weights('saved_models/model/bert-cased-128-04')

# Summarization Model
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
ckpt = 'mrm8488/bert2bert_shared-turkish-summarization'
tokenizer2 = BertTokenizerFast.from_pretrained(ckpt)
model2 = EncoderDecoderModel.from_pretrained(ckpt).to(device)

def generate_summary(text):

   inputs = tokenizer2([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
   input_ids = inputs.input_ids.to(device)
   attention_mask = inputs.attention_mask.to(device)
   output = model2.generate(input_ids, attention_mask=attention_mask)
   return tokenizer2.decode(output[0], skip_special_tokens=True)
   

#data count
count = 211781

def home(request):
    #get random data from database
    random_index = randint(0, count - 1)
    x = FromDataBase.objects.all()[random_index]
    initial_data = {
        'title': x.baslik,
        'complaint': x.metin
    }
    #put random data as initial into the form
    form = ComplaintForm(initial= initial_data)

    return render(request, 'DjangoAPI/index.html', {'form': form})


def classify(request):
    if request.method == 'POST':
        sikayet_metni = request.POST.get('complaint')
        sikayet_baslik = request.POST.get('title')
        total = sikayet_baslik + " " + sikayet_metni
    encoded = encode(total, tokenizer)
    #prediction -> float value
    #classified -> 0 or 1
    prediction =(model.predict(encoded.values()))[0][0]
    classified =(model.predict(encoded.values()) > 0.5).astype("int32")[0][0]

    # For summarization
    summary = {'summary': generate_summary(total)}

    context = {**show_complaint(sikayet_baslik,sikayet_metni),
            **create_graph(prediction, classified),
            **remove_num_and_stopwords(total),
            **remove_puncts(total),
            **lemmatize(total),
            **normalize(total),
            **summary,
            }
    return render(request, 'DjangoAPI/result.html',context)


def show_complaint(sikayet_baslik,sikayet_metni):
    context = {'baslik': sikayet_baslik,
                'metin' : sikayet_metni
                }
    return context

def create_graph(prediction, classified):
    max_value = 6.33
    min_value = -7.47
    normalized = (prediction-min_value)/(max_value-min_value)
    fig = plt.figure(figsize= (4,4))
    height = [normalized*100, (1-normalized)*100]
    bars = ['Network ici','Network disi']
    plt.bar(bars, height, color=('black', 'red'))
    plt.title("Classification")
    plt.ylabel("Probablity (%)")
    plt.xlabel("Network ici   ----------------   Network disi")
    html_graph = mpld3.fig_to_html(fig)
    plt.close()
    if classified == 1:
        classified = 'Network ici'
    else:
        classified = 'Network disi'
    context = {'class': classified,
                'graph': html_graph
             }
    return context
    