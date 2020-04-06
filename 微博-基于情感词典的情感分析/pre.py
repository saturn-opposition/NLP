
# 从mysql数据库中取出文本和对应的标签
import pymysql
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='hjnmysqlmima0930', db='liwenliang', charset='utf8')
cursor = conn.cursor()
cursor.execute("select label,text from comment")
row = cursor.fetchall()
contents = []
id = []
for i in range(len(row)):
    id.append(row[i][0])
    contents.append(row[i][1])
print(len(contents))


import jieba
import string

# 定义预处理函数（去停用词、标点符号等）
def clean(corpus):
    stop = []
    jieba.load_userdict(r"C:\Users\Pluto\Desktop\网上教学\信息计量学\疫情分析项目\数据\搜狗词库\txt\all.txt")
    stop.append("回复")
    with open(r"D:\大创项目\LDA\stopwords\CNstopwords.txt", 'r', encoding='utf-8') as f:
        for lines in f:
            stop.append(lines.strip())
    exclude = set(string.punctuation)  # 标点符号
    clean_corpus = []
    for doc in corpus:
        words = jieba.lcut(doc)
        stop_free = " ".join([i for i in words if (i not in stop) & (i.isalpha())])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        clean_corpus.append(punc_free)
    return clean_corpus


clean_corpus = clean(contents)
print(len(clean_corpus))
for i in range(len(id)):
    cursor.execute("update comment set clean_text = \'" + clean_corpus[i] +"\'" + " where label ="+ str(id[i]))
    conn.commit()
    print(clean_corpus[i])