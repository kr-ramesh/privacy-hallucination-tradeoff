import json
import os
import pickle
import sqlite3

import numpy as np
from rank_bm25 import BM25Okapi

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"


class DocDB:
    def __init__(self, db_path):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Knowledge database {db_path} does not exist. Build one with `python -m factuality_eval.build_db`.")
        self.connection = sqlite3.connect(db_path, check_same_thread=False)

    def get_text_from_title(self, title):
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        cursor.close()
        if len(results) == 1:
            results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
        else:
            results = [{"title": title, "text": row[0]} for row in results]
        assert len(results) > 0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        return results


class Retrieval:
    def __init__(self, db, cache_path, embed_cache_path):
        self.db, self.cache_path, self.embed_cache_path = db, cache_path, embed_cache_path
        self.cache = self._load(cache_path, json.load, "r")
        self.embed_cache = self._load(embed_cache_path, pickle.load, "rb")
        self.add_n, self.add_n_embed = 0, 0

    @staticmethod
    def _load(path, loader, mode):
        try:
            with open(path, mode) as f:
                return loader(f)
        except Exception:
            return {}

    def save_cache(self):
        if self.add_n > 0:
            self.cache.update(self._load(self.cache_path, json.load, "r"))
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
        if self.add_n_embed > 0:
            self.embed_cache.update(self._load(self.embed_cache_path, pickle.load, "rb"))
            with open(self.embed_cache_path, "wb") as f:
                pickle.dump(self.embed_cache, f)

    def get_passages(self, topic, question, k):
        query = topic + " " + question.strip()
        cache_key = topic + "#" + query
        indices_key = cache_key + "#indices"
        if cache_key not in self.cache or indices_key not in self.cache:
            passages = self.db.get_text_from_title(topic)
            if topic not in self.embed_cache:
                self.embed_cache[topic] = BM25Okapi([psg["text"].replace("<s>", "").replace("</s>", "").split() for psg in passages])
                self.add_n_embed += 1
            indices = np.argsort(-self.embed_cache[topic].get_scores(query.split()))[:k]
            self.cache[cache_key], self.cache[indices_key] = [passages[i] for i in indices], indices.tolist()
            self.add_n += 1
        return self.cache[cache_key], self.cache[indices_key]
