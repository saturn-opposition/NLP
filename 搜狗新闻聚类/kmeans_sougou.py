# 从mysql数据库中取出文本和对应的标签
import pymysql
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='hjnmysqlmima0930', db='sougou', charset='utf8')
cursor = conn.cursor()
cursor.execute("select id,content from cluster")
row = cursor.fetchall()
contents = []
id = []
for i in range(len(row)):
    id.append(row[i][0])
    contents.append(row[i][1].replace(' ',''))
print(len(contents))


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

# 对文本数据进行预处理
norm_corpus = clean(contents)
print(norm_corpus[0])

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
bow_vectorizer, bow_features = bow_extractor(norm_corpus)
bow_features_names = bow_vectorizer.get_feature_names()
# 对训练数据/测试数据使用tf-idf模型，将文档转换为向量
tfidf_vectorizer, tfidf_features = tfidf_extractor(norm_corpus)
tfidf_features_names = tfidf_vectorizer.get_feature_names()

from sklearn.cluster import KMeans
def k_means(feature,num_cluster):
    km = KMeans(n_clusters=num_cluster,max_iter=1000)
    km.fit(feature)
    clusters = km.labels_
    return km,clusters

num_cluster = 3

km_bow_model,clusters_bow = k_means(feature=bow_features,num_cluster=num_cluster)
km_tfidf_model,clusters_tfidf = k_means(feature=tfidf_features,num_cluster=num_cluster)

#得出每个聚类的文本数量
from collections import Counter
print("词袋模型+KMeans模型：")
c_bow = Counter(clusters_bow)
print(c_bow.items())
print("tfidf模型+KMeans模型：")
c_tfidf = Counter(clusters_tfidf)
print(c_tfidf.items())

#将聚类结果存入mysql数据库中
for i in range(len(clusters_bow)):
    cursor.execute("update cluster set bow_cluster = \'"+str(clusters_bow[i])+"\' where id =\'" +str(id[i])+"\'")
    conn.commit()

for i in range(len(clusters_tfidf)):
    cursor.execute("update cluster set tfidf_cluster = \'"+str(clusters_tfidf[i])+"\' where id =\'" +str(id[i])+"\'")
    conn.commit()

def get_cluster_feature(clustering_model,features_names,num_cluster,topn_features):
    cluster_details = {}
    ordered_centroids = clustering_model.cluster_centers_.argsort()[:,::-1]
    for cluster_num in range(num_cluster):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [features_names[index] for index in ordered_centroids[cluster_num,:topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features
    return cluster_details
print("\n输出词袋向量+KMeans模型的特征：")
cluster_details = get_cluster_feature(km_bow_model,features_names=bow_features_names,num_cluster=num_cluster,topn_features=6)
for k,v in cluster_details.items():
    print(k)
    print(v)
    print("***********************")
print("\n输出tfidf向量+KMeans模型的特征：")
cluster_details = get_cluster_feature(km_tfidf_model,features_names=tfidf_features_names,num_cluster=num_cluster,topn_features=6)
for k,v in cluster_details.items():
    print(k)
    print(v)
    print("***********************")
from sklearn.externals import joblib
joblib.dump(km_bow_model, 'km_bow.pkl')
joblib.dump(km_tfidf_model, 'km_tfidf.pkl')



