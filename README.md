# NLP
1. LDA主题提取
运用gensim构造LDA模型，在没有对生成的dictionary进行过滤时，dictionary较大，将文档表示为向量的维度太高，调整不同主题个数时得到的困惑度曲线始终呈上升趋势。对idctionary进行过滤后，维度降低，困惑度曲线随主题个数增加而下降，可以选出合适的主题个数。
![困惑度曲线](https://github.com/saturn-opposition/NLP/blob/master/LDA%E4%B8%BB%E9%A2%98%E6%8F%90%E5%8F%96/%E7%96%AB%E6%83%85%E7%9B%B8%E5%85%B3%E5%BE%AE%E5%8D%9A%E4%B8%BB%E9%A2%98%E6%8F%90%E5%8F%96/gensim_lda/perplexityTrend.png)

2. 中文幽默类型
<br>尝试使用卷积神经网络对中文笑话进行分类，可能是特征处理的原因，也可能是这种网络不适用于文本分析，模型效果不佳。

3. 微博情感分析
<br>gensim.word2vec构建词向量模型，sklearn随机森林分类器、XGBoost分类器

4. 搜狗新闻分类
<br>预处理：分词、去停词
<br>划分训练集、测试集（7：3）
<br>特征提取：词袋模型、TFIDF模型、平均词向量模型、TFIDF加权平均词向量模型
<br>模型训练：多项式朴素贝叶斯模型、支持向量机模型
<br>表现：最佳80%精确度

5. 搜狗新闻聚类--Kmeans

6. 电影评论极性分析

7. 基于情感词典的情感分析
<br>对微博进行情感分析，使用的情感词典在文件夹中。先使用pre.py对文本进行预处理，再使用DictSentiment.py计算情感得分，最后用draw.py画出情感得分分布的饼状图和随时间变化的折线图。

