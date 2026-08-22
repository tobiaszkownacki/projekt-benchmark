import os
import psycopg

from shared.interfaces.database_connector import DatabaseConnector


class PostGresConnector(DatabaseConnector):

    def __init__(self) -> None:
        self.host = os.environ.get("POSTGRES_HOST", "postgres")
        self.port = os.environ.get("POSTGRES_PORT")
        self.dbname = os.environ.get("POSTGRES_DB")
        self.user = os.environ.get("POSTGRES_USER")
        self.password = os.environ.get("POSTGRES_PASSWORD")
        self.conn = None

    def __enter__(self) -> "PostGresConnector":
        self.conn = psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.conn is None:
            return
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close()
            self.conn = None

    def execute(self, query: str, params: tuple = ()) -> psycopg.Cursor:
        return self.conn.execute(query, params)