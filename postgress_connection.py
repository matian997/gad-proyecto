from psycopg2 import connect


def getConnection():
    return connect(
        dbname="test_db",
        user="root",
        host="localhost",
        password="root",
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )
