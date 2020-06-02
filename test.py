# test
import pandas as pd
import tensorflow.keras as keras
from keras.models import load_model
import jieba.posseg as pseg
import numpy as np

model = load_model('my_model.h5')


# 建立字典
MAX_SEQUENCE_LENGTH = 10
MAX_NUM_WORDS = 10000
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=MAX_NUM_WORDS)

# Label字典
label_to_index = {
    'n': 0,
    'y': 1,
}


def jieba_tokenizer(text):
    words = pseg.cut(text)
    words = [w for w, t in words]
    return words


def preprocessing(test):
    test['title1_tokenized'] = test['title'].apply(jieba_tokenizer)
    x_test = tokenizer.texts_to_sequences(test['title1_tokenized'])
    x_test = keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=MAX_SEQUENCE_LENGTH)
    return test


text = "不要要記帳"
data = {"title": text}
test = pd.DataFrame(data, index=[0])
test = preprocessing(test)


predictions = model.predict([x_test])


index_to_label = {v: k for k, v in label_to_index.items()}

test['Category'] = [index_to_label[idx]
                    for idx in np.argmax(predictions, axis=1)]
submission = test.loc[:, ['title', 'Category']]

submission.columns = ['title', 'Category']

print(submission)
