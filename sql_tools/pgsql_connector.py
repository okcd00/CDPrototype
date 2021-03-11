# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : pgsql_connector.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-03-11
#   desc     : https://www.postgresql.org/docs/9.4/datatype-json.html
#              https://www.postgresql.org/docs/current/functions-json.html
# ==========================================================================
from common_utils import *
from psycopg2.extras import Json


class PgSqlConnector(object):
    ID = "DarkBase"  # from kdr2

    def __init__(self):
        self.conn = None
        self.table_name = "test_json_table"

        self.cur = self.connect()

    def connect(self, connect_params=None):
        """
        - *dbname*: the database name
        - *database*: the database name (only as keyword argument)
        - *user*: user name used to authenticate
        - *password*: password used to authenticate
        - *host*: database host address (defaults to UNIX socket if not provided)
        - *port*: connection port number (defaults to 5432 if not provided)
        """
        if connect_params is not None:
            self.conn = psycopg2.connect(**connect_params)
        else:
            self.conn = psycopg2.connect(
                dbname="testdb",
                user="postgres",
                password="26pHg2bCwLcz",
                host="127.0.0.1",
                port="35432",
                async=True,
            )
        self.cur = self.conn.cursor()
        return self.cur

    def exec(self, sql_str):
        try:
            self.conn.execute(sql_str)
            return self.cur.fetchall()
        except psycopg2.Error as e:
            self.conn.rollback()
            print(str(e))
            print("Automatically Rollback.")

    def select_table(self, table_name='json_data'):
        self.table_name = table_name

    def create_table(self, table_name='json_data'):
        self.select_table(table_name=table_name)
        create_sql = """
            CREATE TABLE {table_name} (
            ID serial NOT NULL PRIMARY KEY,
            info json NOT NULL
            );""".format(table_name=table_name)
        return create_sql

    def insert_data(self, json_list):
        dumps_list = ['(\'{}\')'.format(json.dumps(sample).replace('\"', '\''))
                      for sample in json_list]
        formatted_json_list = ',\n'.join(dumps_list)
        insert_sql = """
        INSERT INTO {table_name} (info)
        VALUES {json_list};
        """.format(
            table_name=self.table_name,
            json_list=formatted_json_list)
        return insert_sql

    def filter_data(self, select_list, condition_list):
        """

        :param select_list: [
            "info ->> 'customer' AS customer",
            "(info -> 'items' ->> 'qty')::INTEGER",
            "(info -> 'items')::jsonb ? 'categories'"
        ]
        :param condition_list: [
            "info -> 'items' ->> 'product' LIKE 'Toy%'",
            "CAST (info -> 'items' ->> 'qty' AS INTEGER) >= 1",
        ]
        :return:
        """

        formatted_select_list = ',\n'.join(
            select_list)
        formatted_condition_list = '\nAND\n'.join(
            ['({})'.format(cd) for cd in condition_list])
        filter_sql = """
        SELECT
           {formatted_select}
        FROM
           {table_name}
        WHERE
           {formatted_condition}
        """.format(
            formatted_select=formatted_select_list,
            table_name=self.table_name,
            formatted_condition=formatted_condition_list,
        )
        return filter_sql


if __name__ == "__main__":
    psc = PgSqlConnector()
    print(psc.exec(psc.filter_data(
        select_list=[
            "info ->> 'customer' AS customer",
            "(info -> 'items' ->> 'qty')::INTEGER",
            "(info -> 'items')::jsonb ? 'categories'"
        ],
        condition_list=[
            "info -> 'items' ->> 'product' LIKE 'Toy%'",
            "CAST (info -> 'items' ->> 'qty' AS INTEGER) >= 1",
        ]
    )))
