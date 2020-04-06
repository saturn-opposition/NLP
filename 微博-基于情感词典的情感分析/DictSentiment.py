# -*- coding:utf8 -*-

from collections import defaultdict
import os
import re
import jieba
import codecs
import sys
import chardet
import matplotlib.pyplot as plt
import pymysql

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='hjnmysqlmima0930', db='liwenliang')
cursor = conn.cursor()



def load_data():
    cursor.execute("select label,clean_text from comment")
    row = cursor.fetchall()
    ids = []
    contents = []
    for i in range(len(row)):
        ids.append(row[i][0])
        contents.append(row[i][1])
    print(len(ids))
    print(len(contents))
    return ids,contents

# 导入情感词、否定词、程度副词得分词典
def words():
    #情感词
    senList = []
    with open('BosonNLP_sentiment_score.txt','r',encoding="utf-8") as f:
        for line in f:
            senList.append(line.strip('\n'))
    senDict = defaultdict()
    for s in senList:
        senDict[s.split(' ')[0]] = s.split(' ')[1]
    #否定词
    notList = []
    with open('denial_dict.txt','r',encoding="utf-8") as f:
        for line in f:
            notList.append(line.strip('\n'))
    #程度副词
    degreeList = []
    with open('adverb_dict.txt','r',encoding="utf-8") as f:
        for line in f:
            degreeList.append(line.strip('\n'))
    degreeDict = defaultdict()
    for d in degreeList:
        degreeDict[d.split('  ')[0]] = d.split('  ')[1]

    return senDict,notList,degreeDict




# 见文本文档  根据情感定位  获得句子相关得分
def classifyWords(wordDict,senDict,notList,degreeDict):

    senWord = defaultdict()
    notWord = defaultdict()
    degreeWord = defaultdict()
    for word in wordDict.keys():
        if word in senDict.keys() and word not in notList and word not in degreeDict.keys():
            senWord[wordDict[word]] = senDict[word]
        elif word in notList and word not in degreeDict.keys():
            notWord[wordDict[word]] = -1
        elif word in degreeDict.keys():
            degreeWord[wordDict[word]] = degreeDict[word]
    return senWord, notWord, degreeWord


#计算句子得分  见程序文档
def scoreSent(senWord, notWord, degreeWord, segResult):
    W = 1
    score = 0
    # 存所有情感词的位置的列表
    senLoc = list(senWord.keys())
    notLoc = list(notWord.keys())
    degreeLoc = list(degreeWord.keys())
    senloc = -1
    # notloc = -1
    # degreeloc = -1
    # 遍历句中所有单词segResult，i为单词绝对位置
    for i in range(0, len(segResult)):
        # 如果该词为情感词
        if i in senLoc:
            # loc为情感词位置列表的序号
            senloc += 1
            # 直接添加该情感词分数
            score += W * float(senWord[i])
            # print "score = %f" % score
            if senloc < len(senLoc) - 1:
                # 判断该情感词与下一情感词之间是否有否定词或程度副词
                # j为绝对位置
                for j in range(senLoc[senloc], senLoc[senloc + 1]):
                    # 如果有否定词
                    if j in notLoc:
                        W *= -1
                    # 如果有程度副词
                    elif j in degreeLoc:
                        W *= float(degreeWord[j])
        # i定位至下一个情感词
        if senloc < len(senLoc) - 1:
            i = senLoc[senloc + 1]
    return score


#列表 转 字典
def listToDist(wordlist):
    data={}
    for x in range(0, len(wordlist)):
        data[wordlist[x]]=x
    return data

#绘图相关
def runplt():
    plt.figure()
    plt.title('test')
    plt.xlabel('x')
    plt.ylabel('y')
    #这里定义了  图的长度 比如 2000条数据 就要 写 0,2000
    plt.axis([0,1000,-10,10])
    plt.grid(True)
    return plt




#主题从这里开始 上边全是方法


#获取要计算情感得分的数据
ids,contents = load_data()


score_var=[]


#获取 本地的情感词 否定词 程度副词
words_vaule=words()

#循环 读取 contents里的内容
for i in range(len(contents)):
    datafen=contents[i].split(' ')
    #列表转字典
    datafen_dist=listToDist(datafen)
    #通过classifyWords函数 获取句子的 情感词 否定词 程度副词 相关分值
    data_1=classifyWords(datafen_dist,words_vaule[0],words_vaule[1],words_vaule[2])
    # 通过scoreSent 计算 最后句子得分
    data_2=scoreSent(data_1[0],data_1[1],data_1[2],contents[i].split(' '))
    # 将得分保存在score_var 以列表的形式
    score_var.append(data_2)
    cursor.execute("update comment set score = " + str(data_2) +  " where label =\'" + str(ids[i]) + "\'")
    conn.commit()
    #打印句子得分
    print(str(data_2))

#对所有文本得分进行倒序排列
score_var.sort(reverse=True)

#计算一个index 值 存 1~ 所有句子长度 以便于绘图
index=[]
for x in range(0,len(score_var)):
    index.append(x+1)

#初始化绘图
plt=runplt()
#带入参数
plt.plot(index,score_var,'r.')
#显示绘图
plt.show()