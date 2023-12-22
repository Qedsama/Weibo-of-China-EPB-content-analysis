import pandas as pd
import numpy as np
import tensorflow as tf
import jieba
import string
import re
import gensim

YEAR ="2019"
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

def classify_text(model, texts, vocab, default_vector):
    preprocessed_inputs = [] 
    for text in texts:
        preprocessed_inputs.append(preprocess_text(text[1]['微博正文'], vocab, default_vector))
    preprocessed_input = np.vstack(preprocessed_inputs)  
    prediction = model.predict(preprocessed_input)
    prediction = [1 if value > 0.5 else 0 for value in prediction]
    return prediction


if __name__ == "__main__":

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(r'wiki_word2vec_50.bin', binary=True)
    vocab, default_vector = generate_vocab_and_vectors(word2vec_model)
    model = tf.keras.models.load_model('model.h5')
    df = pd.read_csv('./data/merged_data_'+YEAR+'.csv', encoding='utf-8')
    transferred_data = df[df['微博正文'].str.contains('转发内容')]
    transferred_data.to_csv('./results/transferred_data_'+YEAR+'.csv', mode='a', header=False, index=False, encoding='utf-8')
    df = df[~df['微博正文'].str.contains('转发内容')]
    results=classify_text(model, df.iterrows(), vocab, default_vector)
    df['分类结果'] = results
    df = df[['分类结果'] + [col for col in df.columns if col != '分类结果']]
    df.to_csv('./results/result_'+YEAR+'.csv', index=False, encoding='utf-8')
