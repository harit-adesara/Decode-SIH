import os
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

load_dotenv()

DB_URI=os.getenv("DB_URI")

pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={"autocommit": True, "prepare_threshold": 0},
    open=True,
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()