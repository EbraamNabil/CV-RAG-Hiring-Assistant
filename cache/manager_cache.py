import hashlib
import json
import os


CACHE_DIR = "cache"

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


def text_hash(text):

    return hashlib.md5(

        text.encode("utf-8")

    ).hexdigest()
    
def file_hash(file_path):

    sha = hashlib.sha256()

    with open(file_path, "rb") as f:

        while chunk := f.read(8192):

            sha.update(chunk)

    return sha.hexdigest()    


def cache_path(key):

    return os.path.join(

        CACHE_DIR,

        f"{key}.json"

    )


def cache_exists(key):

    return os.path.exists(

        cache_path(key)
    )


def load_cache(key):

    with open(

        cache_path(key),

        encoding="utf-8"

    ) as f:

        return json.load(f)


def save_cache(key, data):

    with open(

        cache_path(key),

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            data,

            f,

            indent=2,

            ensure_ascii=False
        )