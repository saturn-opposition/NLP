from gensim import corpora, models
import pymysql
import random
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='hjnmysqlmima0930', db='sougou', charset='utf8')
cursor = conn.cursor()
cursor.execute("select clean_content from data where label = '运动' order by rand() limit 2000")
row = cursor.fetchall()
sport_contents = []
for i in range(len(row)):
    sport_contents.append(row[i][0].split(' '))
print(len(sport_contents))
print(sport_contents[0])

cursor.execute("select clean_content from data where label = '女性' order by rand() limit 2000")
row = cursor.fetchall()
female_contents = []
for i in range(len(row)):
    female_contents.append(row[i][0].split(' '))
print(len(female_contents))
print(female_contents[0])

cursor.execute("select clean_content from data where label = '游戏' order by rand() limit 2000")
row = cursor.fetchall()
game_contents = []
for i in range(len(row)):
    game_contents.append(row[i][0].split(' '))
print(len(game_contents))
print(game_contents[0])

cursor.execute("select clean_content from data where label = '理财' order by rand() limit 2000")
row = cursor.fetchall()
finance_contents = []
for i in range(len(row)):
    finance_contents.append(row[i][0].split(' '))
print(len(finance_contents))
print(finance_contents[0])

cursor.execute("select clean_content from data where label = '汽车' order by rand() limit 2000")
row = cursor.fetchall()
car_contents = []
for i in range(len(row)):
    car_contents.append(row[i][0].split(' '))
print(len(car_contents))
print(car_contents[0])

contents = sport_contents + female_contents + game_contents + finance_contents + car_contents
random.shuffle(contents)

from gensim.models import TfidfModel
from gensim.sklearn_api import TfIdfTransformer
def ldamodel(num_topics):

    dictionary = corpora.Dictionary(contents)
    dictionary.filter_extremes(10,0.8,keep_n=5000)
    dictionary.compactify()
    corpus = [dictionary.doc2bow(text) for text in
              contents]  # corpus里面的存储格式（0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)
    # print(corpus)
    corpora.MmCorpus.serialize('corpus.mm', corpus)
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda = models.LdaModel(corpus=corpus_tfidf, id2word=dictionary, random_state=1,
                          num_topics=num_topics)  # random_state 等价于随机种子的random.seed()，使每次产生的主题一致

    topic_list = lda.print_topics(num_topics, 10)
    # print("主题的单词分布为：\n")
    # for topic in topic_list:
    #     print(topic)
    return lda, dictionary

import math
def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    print('the info of this ldamodel: \n')
    print('num of topics: %s' % num_topics)
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = []
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = []
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0  # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0
        for word_id, num in dict(doc).items():
            prob_word = 0.0
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic * prob_topic_word
            prob_doc += math.log(prob_word)  # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum / testset_word_num)  # perplexity = exp(-sum(p(d)/sum(Nd))
    print("模型困惑度的值为 : %s" % prep)
    return prep


from gensim import corpora, models
import matplotlib.pyplot as plt

def graph_draw(topic, perplexity):  # 做主题数与困惑度的折线图
    x = topic
    y = perplexity
    plt.plot(x, y, color="red", linewidth=2)
    plt.xlabel("Number of Topic")
    plt.ylabel("Perplexity")
    plt.show()



a = range(2, 20, 2)  # 主题个数
p = []
for num_topics in a:
    lda, dictionary = ldamodel(num_topics)
    corpus = corpora.MmCorpus('corpus.mm')
    testset = []
    for c in range(int(corpus.num_docs / 100)):  # 如何抽取训练集
        testset.append(corpus[c * 100])
    prep = perplexity(lda, testset, dictionary, len(dictionary.keys()), num_topics)
    p.append(prep)

graph_draw(a, p)