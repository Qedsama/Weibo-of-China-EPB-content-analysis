import numpy as np
import tensorflow as tf
import gensim
import jieba
import string
import re
#清洗：
#分词：
#词->词向量：
model = gensim.models.KeyedVectors.load_word2vec_format(r'../bin/wiki_word2vec_50.bin', binary=True)
def clean_text(text):
    # 去掉标点和空格
    translator = str.maketrans('', '', string.punctuation + ' ')
    text = text.translate(translator)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return text
def convert_to_npy(input_file, output_prefix):
    # 读取文本文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析文本数据
    labels = []
    texts = []
    for line in lines:
        parts = line.strip().split(' ',1)
        if len(parts) < 2:
            continue
        labels.append(int(parts[0]))
        line_list = jieba.lcut(clean_text(parts[1]))
        line = ' '.join(line_list)
        # 拆分句子为单词
        text = line.split(' ')
        if len(text) > 64:
            text = text[:64]
        elif len(text) < 64:
            text += ['我'] * (64 - len(text))
        texts.append(text)
    return labels,texts

if __name__ == "__main__":
    y_test,x_test=convert_to_npy(r"../model/test.txt", "test")
    y_train,x_train=convert_to_npy(r"../model/train.txt", "train")
    y_val,x_val=convert_to_npy(r"../model/validation.txt", "validation")
    for i, sen in enumerate(x_train):
        for j, word in enumerate(sen):
            try:
                sen[j] = model[word]
            except KeyError:
                sen[j] = model['我']
    for i, sen in enumerate(x_test):
        for j, word in enumerate(sen):
            try:
                sen[j] = model[word]
            except KeyError:
                sen[j] = model['我']
    for i, sen in enumerate(x_val):
        for j, word in enumerate(sen):
            try:
                sen[j] = model[word]
            except KeyError:
                sen[j] = model['我']

    #转换为numpy数组
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_val = np.array(x_val)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 50)),  # 输入形状为 (batch_size, None, 50)
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(momentum=0.9,epsilon=0.001),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=13, batch_size=64, validation_data=(x_val, y_val))
    model.save('model.h5')

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
