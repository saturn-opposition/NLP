import pymysql
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['font.family'] = 'Kaiti'
plt.rcParams['axes.unicode_minus'] = False
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='hjnmysqlmima0930', db='liwenliang')
cursor = conn.cursor()
cursor.execute("select score,publish_time from comment")
row = cursor.fetchall()
scores = []
time = []
score_dict = {}
count_dict = {}
positive = 0
negative = 0
neutral = 0
for i in range(len(row)):
        scores.append(row[i][0])
        time.append(row[i][1])
        if int(row[i][0]) > 0:
            positive = positive + 1
        elif int(row[i][0]) < 0 :
            negative = negative + 1
        else:
            neutral = neutral + 1
        if row[i][1] in score_dict.keys():
            score_dict[row[i][1]] = score_dict[row[i][1]] + row[i][0]
            count_dict[row[i][1]] = count_dict[row[i][1]] + 1
        else:
            score_dict[row[i][1]] = row[i][0]
            count_dict[row[i][1]] = 1


labels = ['积极','消极','中性']
sizes = [positive,negative,neutral]
explode = (0,0,0)
plt.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=150)
plt.title("饼图示例-评论情感分数")
plt.show()


# xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in time]
print(len(scores))
for k,v in score_dict.items():
    score_dict[k] = v / count_dict[k]

date = []
score = []
for k in sorted(score_dict):
    date.append(k)
    score.append(score_dict[k])

fig = plt.figure()
plt.plot(date,score)
import matplotlib as mpl
ax=plt.gca()
date_format=mpl.dates.DateFormatter('%Y-%m-%d')#设定显示的格式形式
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))#设定坐标轴的显示的刻度间隔
ax.xaxis.set_major_formatter(date_format)#设定x轴主要格式
fig.autofmt_xdate()
plt.xlabel('时间')
plt.ylabel('情感得分')
plt.title("情感得分随时间变化图")
plt.show()