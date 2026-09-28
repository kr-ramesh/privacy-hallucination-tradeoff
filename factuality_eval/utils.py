import pickle
import re
import sqlite3


def truncate_to_last_sentence(text):
    match = re.search(r"^.*[.!?]", text, flags=re.S)
    return match.group(0) if match else text


def save_to_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def titles_in_db(db_file, titles):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    exists = []
    for title in titles:
        cursor.execute("SELECT * FROM documents WHERE title = ?", (title,))
        exists.append(bool(cursor.fetchall()))
    conn.close()
    return exists
