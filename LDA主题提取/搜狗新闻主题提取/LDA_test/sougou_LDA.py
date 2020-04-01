import pymysql
import joblib
from time import time
import random
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='hjnmysqlmima0930', db='sougou', charset='utf8')
cursor = conn.cursor()
cursor.execute("select clean_content from data where label = '运动' order by rand() limit 2000")
row = cursor.fetchall()
sport_contents = []
for i in range(len(row)):
    sport_contents.append(row[i][0])
print(len(sport_contents))
print(sport_contents[0])

cursor.execute("select clean_content from data where label = '女性' order by rand() limit 2000")
row = cursor.fetchall()
female_contents = []
for i in range(len(row)):
    female_contents.append(row[i][0])
print(len(female_contents))
print(female_contents[0])

cursor.execute("select clean_content from data where label = '游戏' order by rand() limit 2000")
row = cursor.fetchall()
game_contents = []
for i in range(len(row)):
    game_contents.append(row[i][0])
print(len(game_contents))
print(game_contents[0])

cursor.execute("select clean_content from data where label = '理财' order by rand() limit 2000")
row = cursor.fetchall()
finance_contents = []
for i in range(len(row)):
    finance_contents.append(row[i][0])
print(len(finance_contents))
print(finance_contents[0])

cursor.execute("select clean_content from data where label = '汽车' order by rand() limit 2000")
row = cursor.fetchall()
car_contents = []
for i in range(len(row)):
    car_contents.append(row[i][0])
print(len(car_contents))
print(car_contents[0])

all_data = sport_contents + female_contents + game_contents + finance_contents + car_contents
random.shuffle(all_data)
#构建TFIDF模型
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=10, norm='l2', smooth_idf=True, use_idf=True, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# tf_vectorizer, tf = tfidf_extractor(all_data)
# joblib.dump(tf_vectorizer, "sougou_tfidf_vectorizer.model")
tf_vectorizer = joblib.load("sougou_tfidf_vectorizer.model")
temp = tf_vectorizer.fit_transform(all_data)
tf = temp[:7000,:]
tf_test = temp[7000:,:]
print(len(tf_vectorizer.get_feature_names()))


from sklearn.decomposition import LatentDirichletAllocation
n_topics = range(2, 10, 1)
perplexityLst = [1.0]*len(n_topics)

#训练LDA并打印训练时间
lda_models = []
for idx, n_topic in enumerate(n_topics):
    lda = LatentDirichletAllocation(n_components=n_topic,
                                    max_iter=20,
                                    learning_method='batch',
                                    evaluate_every=200,
#                                    perp_tol=0.1, #default
#                                    doc_topic_prior=1/n_topic, #default
#                                    topic_word_prior=1/n_topic, #default
                                    verbose=1)
    t0 = time()
    lda.fit(tf)
    perplexityLst[idx] = lda.perplexity(tf_test)
    # perplexityLst[idx] = perplexity(lda,tf,tf_vectorizer.vocabulary_,len(tf_vectorizer.vocabulary_),n_topic)
    lda_models.append(lda)
    print("# of Topic: %d, " % n_topics[idx])
    print("done in %0.3fs, N_iter %d, " % ((time() - t0), lda.n_iter_))
    print("Perplexity Score %0.3f" % perplexityLst[idx])

#打印最佳模型
best_index = perplexityLst.index(min(perplexityLst))
best_n_topic = n_topics[best_index]
best_model = lda_models[best_index]
print("Best # of Topic: ", best_n_topic)

import matplotlib.pyplot as plt
import os
#绘制不同主题数perplexity的不同
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(n_topics, perplexityLst)
ax.set_xlabel("# of topics")
ax.set_ylabel("Approximate Perplexity")
plt.grid(True)
plt.savefig( 'perplexityTrend.png')
plt.show()

import pyLDAvis.sklearn

data = pyLDAvis.sklearn.prepare(best_model, tf, tf_vectorizer)

pyLDAvis.show(data)#可视化主题模型
