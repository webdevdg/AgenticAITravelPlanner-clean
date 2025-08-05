

# store/redis_store.py
import os
import redis
from langgraph.store.base import BaseStore

class RedisStore(BaseStore):
    def __init__(self, url, namespace="default"):
        self.client = redis.from_url(url)
        self.ns = namespace + ":"

    @classmethod
    def from_url(cls, url, namespace="default"):
        return cls(url, namespace)

    def _key(self, thread_id, key):
        return f"{self.ns}{thread_id}:{key}"

    def get(self, thread_id, key, default=None):
        v = self.client.get(self._key(thread_id, key))
        return v.decode() if v else default

    def put(self, thread_id, key, value):
        # store everything as strings
        self.client.set(self._key(thread_id, key), str(value))

    def delete(self, thread_id, key):
        self.client.delete(self._key(thread_id, key))

    def list_keys(self, thread_id):
        pattern = f"{self.ns}{thread_id}:*"
        # returns full keys, so strip namespace/thread_id
        return [k.decode().split(":", 2)[-1] for k in self.client.keys(pattern)]

    # ← implement these two to satisfy BaseStore’s abstract interface
    def batch(self, thread_id, mapping: dict):
        """
        Synchronously write multiple key→value pairs at once.
        """
        pipeline = self.client.pipeline()
        for k, v in mapping.items():
            pipeline.set(self._key(thread_id, k), str(v))
        pipeline.execute()

    async def abatch(self, thread_id, mapping: dict):
        """
        Async version of batch(). For now, just run batch() in executor.
        """
        # If you don’t use async, you can just call the sync version:
        return self.batch(thread_id, mapping)
