import pandas as pd
import numpy as np
import tensorflow as tf
import jieba
import string
import re
import gensim
import concurrent.futures
import csv

def generate_vocab_and_vectors(model):
    vocab = {}
    default_vector = np.zeros_like(model['我']) 

    for index, word in enumerate(model.index_to_key):
        vocab[word] = model[word]

    return vocab, default_vector

def clean_text(text):
    translator = str.maketrans('', '', string.punctuation + ' ')
    text = text.translate(translator)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return text

def preprocess_text(text, vocab, default_vector):
    cleaned_text = clean_text(text)
    segmented_text = jieba.lcut(cleaned_text)
    input_text = segmented_text[:64] + ['我'] * (64 - len(segmented_text))
    
    input_vector = [vocab.get(word, default_vector) for word in input_text]

    input_array = np.array(input_vector)
    return np.expand_dims(input_array, axis=0)

def classify_text(model, text, vocab, default_vector):
    preprocessed_input = preprocess_text(text, vocab, default_vector)
    prediction = model.predict(preprocessed_input)
    return 1 if prediction > 0.5 else 0

def process_weibo_text(args):
    weibo_text, model, vocab, default_vector = args
    return classify_text(model, weibo_text['微博正文'], vocab, default_vector)

if __name__ == "__main__":

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(r'../bin/wiki_word2vec_50.bin', binary=True)
    vocab, default_vector = generate_vocab_and_vectors(word2vec_model)


    model = tf.keras.models.load_model('model.h5')


    df = pd.read_csv('../data/merged_data_2015.csv', encoding='utf-8')
    #更换为对应年份的数据


    transferred_data = df[df['微博正文'].str.contains('转发内容')]
    transferred_data.to_csv('./results/transferred_data_2015.csv', mode='a', header=False, index=False, encoding='utf-8')
    #更换为对应年份的数据

    df = df[~df['微博正文'].str.contains('转发内容')]


    with concurrent.futures.ThreadPoolExecutor() as executor:
        num_threads = 4
        results = list(executor.map(process_weibo_text, [(row, model, vocab, default_vector) for _, row in df.iterrows()]))

    df['分类结果'] = results
    df = df[['分类结果'] + [col for col in df.columns if col != '分类结果']]
    df.to_csv('./results/result_2015.csv', index=False, encoding='utf-8')
    #更换为对应年份的数据
