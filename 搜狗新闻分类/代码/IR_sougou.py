# 从mysql数据库中取出文本和对应的标签
import pymysql
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='hjnmysqlmima0930', db='sougou', charset='utf8')
cursor = conn.cursor()
cursor.execute("select label,content from data")
row = cursor.fetchall()
labels = []
contents = []
for i in range(len(row)):
    labels.append(row[i][0])
    contents.append(row[i][1].replace(' ',''))
print(len(labels))
print(len(contents))
print(len(set(labels)))


import jieba
import string

# 定义预处理函数（去停用词、标点符号等）
def clean(corpus):
    stop = []
    with open(r"D:\大创项目\LDA\stopwords\CNstopwords.txt", 'r', encoding='utf-8') as f:
        for lines in f:
            stop.append(lines.strip())
    stop.append(r'\u3000')
    stop.append(r'\ue40c')
    exclude = set(string.punctuation)  # 标点符号
    clean_corpus = []
    for doc in corpus:
        words = jieba.lcut(doc)
        stop_free = " ".join([i for i in words if (i not in stop) & (i.isalpha())])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        clean_corpus.append(punc_free)
    return clean_corpus

#划分训练集、测试集（7：3）
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(contents, labels, test_size=0.3)
# 对划分好的训练集、测试集进行预处理
norm_train_corpus = clean(train_X)
norm_test_corpus = clean(test_X)
print(len(norm_train_corpus))
print(len(norm_test_corpus))
print(norm_train_corpus[0])

# 词袋模型
from sklearn.feature_extraction.text import CountVectorizer
def bow_extractor(corpus, ngram_range=(1, 1)):  # ngram_range参数是指，将前后多少个词组合，构造新的词袋标签
    vectorizer = CountVectorizer(min_df=10, ngram_range=ngram_range)  # min_df是指最小出现多少次的也算入词袋，本实验中因数据量较大，设为10
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


# TF-IDF模型
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=10, norm='l2', smooth_idf=True, use_idf=True, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# 对训练数据/测试数据使用词袋模型，将文档转换为词袋向量
bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
# 使用训练数据生成的词向量模型，将测试集转换为词袋向量
bow_test_features = bow_vectorizer.transform(norm_test_corpus)
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)


from sklearn import metrics
import numpy as np
# 作出混淆矩阵
def get_metrics(true_labels, predicted_labels):
    print('Accuracy:' + str(np.round(metrics.accuracy_score(true_labels, predicted_labels), 2)))
    print('Percision:' + str(np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 2)))
    print('Recall:' + str(np.round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 2)))
    print('F1 Score:' + str(np.round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 2)))

# 定义函数使用机器学习算法训练模型
def train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels):
    # 使用分类器训练数据
    model = classifier.fit(train_features, train_labels)
    # 使用训练好的模型对测试集进行预测
    predictions = classifier.predict(test_features)
    # 对模型表现进行评估
    get_metrics(true_labels=test_labels, predicted_labels=predictions)
    return predictions,model

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
mnb = MultinomialNB()
knn=KNeighborsClassifier()

import pandas as pd
mnb_bow_predictions,mnb_bow_model = train_predict_evaluate_model(classifier=mnb, train_features=bow_train_features,
                                                   train_labels=train_Y, test_features=bow_test_features,
                                                   test_labels=test_Y)
cm = metrics.confusion_matrix(test_Y, mnb_bow_predictions,labels=['运动','女性','理财','校园','商务','音乐','汽车','交易商','产品','互联网','游戏','娱乐','景区','信用卡'])
print(pd.DataFrame(cm, index=range(0, 14), columns=range(0, 14)))

mnb_tfidf_predictions,mnb_tfidf_model = train_predict_evaluate_model(classifier=mnb, train_features=tfidf_train_features,
                                                     train_labels=train_Y, test_features=tfidf_test_features,
                                                     test_labels=test_Y)
cm = metrics.confusion_matrix(test_Y, mnb_tfidf_predictions,labels=['运动','女性','理财','校园','商务','音乐','汽车','交易商','产品','互联网','游戏','娱乐','景区','信用卡'])
print(pd.DataFrame(cm, index=range(0, 14), columns=range(0, 14)))

knn_bow_predictions,knn_bow_model = train_predict_evaluate_model(classifier=knn, train_features=bow_train_features,
                                                   train_labels=train_Y, test_features=bow_test_features,
                                                   test_labels=test_Y)
cm = metrics.confusion_matrix(test_Y, knn_bow_predictions,labels=['运动','女性','理财','校园','商务','音乐','汽车','交易商','产品','互联网','游戏','娱乐','景区','信用卡'])
print(pd.DataFrame(cm, index=range(0, 14), columns=range(0, 14)))

knn_tfidf_predictions,knn_tfidf_model = train_predict_evaluate_model(classifier=knn, train_features=tfidf_train_features,
                                                     train_labels=train_Y, test_features=tfidf_test_features,
                                                     test_labels=test_Y)
cm = metrics.confusion_matrix(test_Y, knn_tfidf_predictions,labels=['运动','女性','理财','校园','商务','音乐','汽车','交易商','产品','互联网','游戏','娱乐','景区','信用卡'])
print(pd.DataFrame(cm, index=range(0, 14), columns=range(0, 14)))

from sklearn.metrics import classification_report
report = classification_report(mnb_bow_predictions,test_Y)
print(report)

report = classification_report(mnb_tfidf_predictions,test_Y)
print(report)

report = classification_report(knn_bow_predictions,test_Y)
print(report)

report = classification_report(knn_tfidf_predictions,test_Y)
print(report)

from sklearn.externals import joblib
joblib.dump(mnb_bow_model, 'mnb_bow.pkl')
joblib.dump(mnb_tfidf_model, 'mnb_tfidf.pkl')
joblib.dump(knn_bow_model, 'knn_bow.pkl')
joblib.dump(knn_tfidf_model, 'knn_tfidf.pkl')

import pickle
tfidf_path = 'tfidf_feature.pkl'
with open(tfidf_path, 'wb') as fw:
    pickle.dump(tfidf_vectorizer, fw)

bow_path = 'bow_feature.pkl'
with open(bow_path, 'wb') as fw:
    pickle.dump(bow_vectorizer, fw)