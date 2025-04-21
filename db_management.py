import ssl
import pymysql
import os

db_host=os.environ.get('DB_HOST')
db_user=os.environ.get('DB_USER')
db_password=os.environ.get('DB_PASSWORD')
db_name=os.environ.get('DB_NAME')
db_port=int(os.environ.get('DB_PORT', 3306))

def get_db_connection():
    connection = pymysql.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        db=db_name,
        ssl={
        "cert_reqs": ssl.CERT_NONE,
        "check_hostname": False
        },
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection