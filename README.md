# NLP
1. LDA主题提取
+ 运用gensim构造LDA模型，在没有对生成的dictionary进行过滤时，dictionary较大，将文档表示为向量的维度太高，调整不同主题个数时得到的困惑度曲线始终呈上升趋势。对idctionary进行过滤后，维度降低，困惑度曲线随主题个数增加而下降，可以选出合适的主题个数。
![困惑度曲线](https://github.com/saturn-opposition/NLP/blob/master/LDA%E4%B8%BB%E9%A2%98%E6%8F%90%E5%8F%96/%E7%96%AB%E6%83%85%E7%9B%B8%E5%85%B3%E5%BE%AE%E5%8D%9A%E4%B8%BB%E9%A2%98%E6%8F%90%E5%8F%96/perplexityTrend.png)
