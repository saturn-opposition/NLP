import pymysql
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='hjnmysqlmima0930', db='liwenliang', charset='utf8')
cursor = conn.cursor()
cursor.execute("select id,publish_time from weibo")
row = cursor.fetchall()
publish_t = []
id = []
for i in range(len(row)):
    id.append(row[i][0])
    publish_t.append(row[i][1])

cursor.execute("select id from comment")
row = cursor.fetchall()
id_c = []
print(len(row))
for i in range(64343,len(row)):
    # id_c.append(row[i][0])
    for j in range(len(id)):
        if row[i][0] == id[j]:
            cursor.execute(
                "update comment set publish_time = \'" + publish_t[j] + "\'" + " where id =\'" + row[i][0] + "\'")
            conn.commit()
            print(i)
            break