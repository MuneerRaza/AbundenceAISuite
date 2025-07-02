import os
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv
load_dotenv()


_client = MongoClient(os.getenv("DB_URI"))
checkpointer = MongoDBSaver(_client, db_name='abundance_ai')

def delete_thread(thread_id: str):
    db = _client['abundance_ai']
    checkpoint = db['checkpoints']
    checkpoint_writes = db['checkpoint_writes']
    checkpoint.delete_many({"thread_id": thread_id})
    checkpoint_writes.delete_many({"thread_id": thread_id})