#!/usr/bin/env python
# coding: utf-8

# # 获取文本和标签

# In[33]:



import re
with open(r"D:\NLP\news_sohusite_xml.full\news_sohusite_xml.txt",'r',encoding='ansi') as f:
    raw_data = f.read()     


# In[34]:


patternURL = re.compile(r'<url>(.*?)</url>', re.S)
patternCtt = re.compile(r'<content>(.*?)</content>', re.S)
labels = []
urls = patternURL.findall(raw_data)
contents = patternCtt.findall(raw_data)
for i in range(len(urls)):
    pattern = re.compile(r'http://(.*?)\.',re.S)
    labels.append(pattern.findall(urls[i]))
print(len(labels))
print(len(contents))
print(labels[0])
print(contents[0])


# # 预处理

# In[35]:


import jieba
import gensim
import emoji


# In[36]:



def clean(corpus):
    stopwords = []
    with open(r"D:\大创项目\LDA\stopwords\CNstopwords.txt",'r',encoding='utf8') as f:
        for line in f:
            stopwords.append(line.strip('\n'))
#     print(stopwords)
    clean_corpus = []
    for doc in corpus:
        words_raw = jieba.lcut(emoji.demojize(doc))
        words_clean = []
        for w in words_raw:
            if (w not in stopwords) :
                words_clean.append(w)
        clean_corpus.append(''.join(words_clean))
    return clean_corpus


# In[ ]:





# # 划分训练集，测试集

# In[37]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(contents,labels,test_size=0.3)


# In[ ]:


norm_train_corpus = clean(train_X)
norm_test_corpus = clean(test_X)


# # 特征提取

# In[19]:


#词袋模型
from sklearn.feature_extraction.text import CountVectorizer
def bow_extractor(corpus,ngram_range=(1,1)):     #ngram_range参数是指，将前后多少个词组合，构造新的词袋标签
    vectorizer = CountVectorizer(min_df=1,ngram_range=ngram_range)      #min_df是指最小出现1次的也算入词袋
    features = vectorizer.fit_transform(corpus)
    return vectorizer,features

#TF-IDF模型
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_extractor(corpus,ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(min_df=1,norm='l2',smooth_idf=True,use_idf=True,ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer,features

#平均词向量模型
import numpy as np
def averaged_word_vectors(words,model,vocabulary,num_features):
    feature_vector = np.zeros((num_features,),dtype='float64')
    nwords = 0.
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1
            feature_vector = np.add(feature_vector,model[word])
    if nwords:
        feature_vector = np.divide(feature_vector,nwords)
    return feature_vector
def averaged_word_vectorizer(corpus,model,num_features):
    vocabulary = set(model.wv.index2word)
    features = [averaged_word_vectors(tokenized_sentence,model,vocabulary,num_features) for tokenized_sentence in corpus]
    return np.array(features)

#TF-IDF加权平均词向量模型
def tfidf_wtd_avg_word_vectors(words,tfidf_vector,tfidf_vocabulary,model,num_features):
    #获取所有词的tf-idf权重
    word_tfidfs = [tfidf_vector[0,tfidf_vocabulary.get(word)]if tfidf_vocabulary.get(word) else 0 for word in words]
    #将所得的每个词权重的list建成一个词典
    word_tfidf_map = {word:tfidf_val for word,tfidf_val in zip(words,word_tfidfs)}
    
    feature_vector = np.zeros((num_features,),dtype='float64')
    vocabulary = set(model.wv.index2word)
    wts = 0.
    for word in words:
        if word in vocabulary:
            word_vector = model[word]
            weight_vector = word_tfidf_map[word]*word_vector
            wts = wts + word_tfidf_map[word]
            feature_vector = np.add(feature_vector,weight_vector)
    if wts:
        feature_vector = np.divide(feature_vector,wts)
    return feature_vector

def tfidf_weighted_averaged_word_vectorizer(corpus,tfidf_vectors,tfidf_vocabulary,model,num_features):
    docs_tfidfs = [(doc,doc_tfidf) for doc,doc_tfidf in zip(corpus,tfidf_vectors)]
    features = [tfidf_wtd_avg_word_vectors(tokenized_sentence,tfidf,tfidf_vocabulary,model,num_features) for tokenized_sentence,tfidf in docs_tfidfs]
    return np.array(features)

    


# In[21]:


#对训练数据/测试数据使用词袋模型，将文档转换为词袋向量
bow_vectorizer,bow_train_features = bow_extractor(norm_train_corpus)
#使用训练数据生成的词向量模型，将测试集转换为向量
bow_test_features = bow_vectorizer.transform(norm_test_corpus)

tfidf_vectorizer,tfidf_train_features = tfidf_extractor(norm_train_corpus)
tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

#对文档集中的每个文档进行分句处理
tokenized_train = [jieba.lcut(text) for text in norm_train_corpus]
tokenized_test = [jieba.lcut(text) for text in norm_test_corpus]

#建立word2vec模型，模型返回词汇表中每个词的向量表示，而不是每个文档的向量表示
#因此需要使用平均词向量模型或TF-IDF平均加权词向量模型
model = gensim.models.Word2Vec(tokenized_train,size=100,sample=1e-3)

#平均词向量模型
avg_wv_train_features = averaged_word_vectorizer(corpus=tokenized_train,model=model,num_features=100)
avg_wv_test_features = averaged_word_vectorizer(corpus=tokenized_test,model=model,num_features=100)

#TF-IDF加权平均词向量模型
#别忘了加下划线
vocab = tfidf_vectorizer.vocabulary_

tfidf_wv_train_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_train,tfidf_vectors=tfidf_train_features,tfidf_vocabulary=vocab,model=model,num_features=100)
tfidf_wv_test_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_test,tfidf_vectors=tfidf_test_features,tfidf_vocabulary=vocab,model=model,num_features=100)


# # 评价模型

# In[22]:


from sklearn import metrics
import numpy as np

#作出混淆矩阵
def get_metrics(true_labels,predicted_labels):
    print('Accuracy'+str(np.round(metrics.accuracy_score(true_labels,predicted_labels),2)))
    print('Percision'+str(np.round(metrics.precision_score(true_labels,predicted_labels,average='weighted'),2)))     
    print('Recall'+str(np.round(metrics.recall_score(true_labels,predicted_labels,average='weighted'),2)))
    print('F1 Score'+str(np.round(metrics.f1_score(true_labels,predicted_labels,average='weighted'),2)))
    


# # 定义函数使用机器学习算法训练模型

# In[23]:


def train_predict_evaluate_model(classifier,train_features,train_labels,test_features,test_labels):
    #使用分类器训练数据
    classifier.fit(train_features,train_labels)
    #使用训练好的模型对测试集进行预测
    predictions = classifier.predict(test_features)
    #对模型表现进行评估
    get_metrics(true_labels=test_labels,predicted_labels=predictions)
    return predictions


# # 使用scikit-learn引入机器学习算法模型

# In[24]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge',n_iter_no_change=10)


# In[26]:


mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,train_features=bow_train_features,train_labels=train_Y,test_features=bow_test_features,test_labels=test_Y)


# In[29]:


svm_bow_predictions = train_predict_evaluate_model(svm,bow_train_features,train_Y,bow_test_features,test_Y)


# In[30]:


mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,train_features=tfidf_train_features,train_labels=train_Y,test_features=tfidf_test_features,test_labels=test_Y)


# In[ ]:


import pandas as pd
cm = metrics.confusion_matrix(test_Y,svm_tfidf_predictions)
pd.DataFrame(cm,index=range(0,20),columns=range(0,20))


# In[ ]:




