import jieba.posseg as pseg
import tensorflow.keras as keras
import pandas as pd
import numpy as np
from keras.layers import Embedding, LSTM, concatenate, Dense
from keras import Input
from keras.models import Model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
# 基本參數設置，有幾個分類
NUM_CLASSES = 2

# 在語料庫裡有多少詞彙
MAX_NUM_WORDS = 10000

# 一個標題最長有幾個詞彙
MAX_SEQUENCE_LENGTH = 10

# 一個詞向量的維度
NUM_EMBEDDING_DIM = 256

# LSTM 輸出的向量維度
NUM_LSTM_UNITS = 128

# 建立字典
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


def preprocess():
    train = pd.read_csv('testdata.csv')
    train = pd.DataFrame(train)

    train['title1_tokenized'] = train['title'].apply(jieba_tokenizer)  # 分割語句
    corpus = train['title1_tokenized']
    tokenizer.fit_on_texts(corpus)
    x_train = tokenizer.texts_to_sequences(corpus)  # 將分割後的語句，轉換為數字

    # Zero Padding
    MAX_SEQUENCE_LENGTH = 10
    x_train = keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=MAX_SEQUENCE_LENGTH)

    y_train = train['label'].apply(lambda x: label_to_index[x])
    y_train = np.asarray(y_train).astype('float32')
    y_train = keras.utils.to_categorical(y_train)

    # split val data
    VALIDATION_RATIO = 0.1
    RANDOM_STATE = 9527

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=VALIDATION_RATIO,
        random_state=RANDOM_STATE
    )

    return x_train, x_val, y_train, y_val


def build_model():
    top_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')

    embedding_layer = Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
    top_embedded = embedding_layer(top_input)
    shared_lstm = LSTM(NUM_LSTM_UNITS)
    top_output = shared_lstm(top_embedded)

    dense = Dense(
        units=NUM_CLASSES,
        activation='softmax')
    predictions = dense(top_output)

    model = Model(
        inputs=top_input,
        outputs=predictions)

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def train_model(x_train, x_val, y_train, y_val, model):
    # train model
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    history = model.fit(
        # 輸入是兩個長度為 20 的數字序列
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        # 每個 epoch 完後計算驗證資料集
        # 上的 Loss 以及準確度
        validation_data=(
            x_val,
            y_val
        ),
        # 每個 epoch 隨機調整訓練資料集
        # 裡頭的數據以讓訓練過程更穩定
        shuffle=True
    )

    return model


x_train, x_val, y_train, y_val = preprocess()
model = build_model()
model = train_model(x_train, x_val, y_train, y_val, model)

model.save('my_model.h5')

