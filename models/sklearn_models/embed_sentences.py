import os
import pickle

import numpy as np

import gensim.downloader as api
from gensim.models import KeyedVectors

LOCAL_MODEL_PATH = os.getenv("AITA_GENSIM_MODEL_PATH", "")

GENSIM_WORD_EMBEDDINGS_PATH = "word2vec-google-news-300"
MODEL_FILENAME = GENSIM_WORD_EMBEDDINGS_PATH

AGGREGATION = "sum"

SAMPLE_COMMENT = "NTA - You're not the asshole for hurting everybody."


def main():
    # Getting vector model
    w2v_model = get_model(MODEL_FILENAME)
    embed = get_sentence_embedding(w2v_model, SAMPLE_COMMENT)
    print(embed.shape)


def get_model(filename, default_gensim_path=GENSIM_WORD_EMBEDDINGS_PATH):
    if not LOCAL_MODEL_PATH:
        # If you are not caching gensim in some other folder via pip,
        # just load the model
        return api.load(default_gensim_path)

    filepath = os.path.join(LOCAL_MODEL_PATH, filename)
    print(filepath)

    # todo: cache model to speed up model fetch
    if not os.path.isfile(filepath):
        print(f"downloading model from {default_gensim_path}")
        w2v_model = api.load(default_gensim_path)
        with open(filepath, 'wb') as f:
            pickle.dump(w2v_model, f)
    else:
        print(f"importing model from {filepath}")
        with open(filepath, 'rb') as f:
            w2v_model = pickle.load(f)
    return w2v_model


def get_sentence_embedding(w2v_model, comment_text, aggregation="avg"):
    # print('getting individual token embeddings')
    words = comment_text.split()
    word_vecs = [w2v_model[word] for word in words if word in w2v_model.key_to_index]

    # check length
    for word_vec in word_vecs:
        assert len(word_vec) == 300

    # print('getting sentence embedding')
    word_vec_aggregation = None
    if aggregation == "avg":
        word_vec_aggregation = sum(word_vecs) / len(word_vecs)

    return word_vec_aggregation


if __name__ == "__main__":
    main()
