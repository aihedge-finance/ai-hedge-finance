
import pymysql


class MySQLService:

    def __init__(self, sql_config=None, sqlalchemy_url=None):
        self.sql_config = sql_config
        self.sqlalchemy_url = sqlalchemy_url

    def create_connection(self):
        connection = pymysql.connect(**self.sql_config)
        """
        if self.sql_config:
            connection = pymysql.connect(**self.sql_config)
        elif self.sqlalchemy_url:
            connection = create_engine(
                self.sqlalchemy_url,
                pool_recycle=3600,
                poolclass=NullPool,
                pool_pre_ping=True,
            ).raw_connection()
        """
        connection.set_charset('utf8mb4')
        return connection

    def execute(self, sql, args=None):
        try:
            connection = self.create_connection()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, args)
            last_row_id = cursor.lastrowid
            connection.commit()
            return last_row_id
        finally:
            cursor.close()
            connection.close()

    def execute_fmt(self, sql, args=None):
        return self.execute(sql, args)

    def execute_mulit(self, sql_list):
        try:
            connection = self.create_connection()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            for sql in sql_list:
                cursor.execute(sql)
                rowcount = cursor.rowcount
            connection.commit()
            return rowcount
        finally:
            cursor.close()
            connection.close()

    def executemany(self, sql, values):
        try:
            connection = self.create_connection()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            cursor.executemany(sql, values)
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    def fetchone(self, sql, args=None):
        try:
            connection = self.create_connection()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, args)
            result = cursor.fetchone()
            return result
        finally:
            cursor.close()
            connection.close()

    def fetchall(self, sql, args=None):
        try:
            connection = self.create_connection()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, args)
            result = cursor.fetchall()
            return result
        finally:
            cursor.close()
            connection.close()

    def fetchall_limit(self, sql, args=None):
        try:
            connection = self.create_connection()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, args)
            result = cursor.fetchall()
            cursor.execute(""" SELECT FOUND_ROWS() AS rowcount """)
            row = cursor.fetchone()
            return result, row["rowcount"]
        finally:
            cursor.close()
            connection.close()
