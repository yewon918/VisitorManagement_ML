# https://lucathree.github.io/python/day16/
# 호스트 http://ferrydraw.dothome.co.kr/myadmin

import pymysql

conn = pymysql.connect(
    host= 'localhost',      # 서버 입장에서 DB가 같은 서버에 존재, localhost
    user='ferrydraw',
    password='hsh0729!', db='RTW2020', charset='utf8')
# conn = pymysql.connect(
#     host= 'localhost',      # 서버 입장에서 DB가 같은 서버에 존재, localhost
#     user='root',
#     password='0918', db='RTW2020', charset='utf8')
print('성공')

cur = conn.cursor()

# sql문 실행, fetch
sql = 'SELECT * FROM signeduser_table'
cur.execute(sql)

rows = cur.fetchall()
for row in rows:
    print(row)

# db 연결 종료
conn.close()

