# accuracy:82%
#共14类文本，50000条数据，训练集与测试集比例为7：3
import pymysql
import jieba
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='hjnmysqlmima0930', db='sougou', charset='utf8')
cursor = conn.cursor()
cursor.execute("select label,clean_content from data order by rand() LIMIT 50000")
row = cursor.fetchall()
labels = []
contents = []
for i in range(len(row)):
    labels.append(row[i][0])
    contents.append(row[i][1])

print(len(set(labels)))
print((set(labels)))

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

values = array(labels)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
print(len(onehot_encoded))


from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(contents, onehot_encoded, test_size=0.3)

from gensim.models import Word2Vec
import numpy as np

# model = Word2Vec(contents, size=128, window=5, min_count=5, workers=4)
# model.save("sougou_news.model")

def averaged_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype='float64')
    nwords = 0.
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1
            feature_vector = np.add(feature_vector, model[word])
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [averaged_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in
                corpus]
    return np.array(features)

tokenized_train = [jieba.lcut(text) for text in train_X]
tokenized_test = [jieba.lcut(text) for text in test_X]

# 建立word2vec模型，模型返回词汇表中每个词的向量表示，而不是每个文档的向量表示
# 因此需要使用平均词向量模型或TF-IDF平均加权词向量模型
model = Word2Vec.load("sougou_news.model")

# 平均词向量模型
avg_wv_train_features = averaged_word_vectorizer(corpus=tokenized_train, model=model, num_features=128)
avg_wv_test_features = averaged_word_vectorizer(corpus=tokenized_test, model=model, num_features=128)
avg_wv_train_features = avg_wv_train_features.reshape((avg_wv_train_features.shape[0],avg_wv_train_features.shape[1],1))
avg_wv_test_features = avg_wv_test_features.reshape((avg_wv_test_features.shape[0],avg_wv_test_features.shape[1],1))
print(avg_wv_train_features.shape)
print(avg_wv_test_features.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow.keras.layers import Conv1D ,MaxPooling1D,GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
# embeddind层将正整数（索引值）转换为固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]，该层只能用作模型中的第一层。
# 词汇表大小是max_features,输出词向量的维度是embedding_dims，input_length参数指输入序列的长度
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 20


# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='same',
                 activation='relu',
                 strides=1,
                 input_shape=(128,1)))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='same',
                 activation='relu',
                 strides=1))
# we use max pooling:
# 叠加的时候总是会出现shape不匹配的问题？？？
# https://blog.csdn.net/linxid/article/details/86426506
# GlobalMaxPooling会改变维度！！！！！
model.add(MaxPooling1D())


model.add(Conv1D(filters,
                 kernel_size,
                 padding='same',
                 activation='relu',
                 strides=1))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='same',
                 activation='relu',
                 strides=1))
# # we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(14))
model.add(Activation('sigmoid'))

model.build((None,10,2))
# 使用二分类损失函数
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(avg_wv_train_features, train_Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(avg_wv_test_features, test_Y))


