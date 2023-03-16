# Inspiration taken from https://www.kaggle.com/code/sunruslan/ai-group-jetbrains-internship

# Library imports
import os

import pandas as pd

from sklearn.model_selection import train_test_split

from models.sklearn_models import models_utils

from models.sklearn_models.embed_sentences import get_model
from models.sklearn_models.embed_sentences import get_sentence_embedding

SEED = 42
BATCH_SIZE = 32
NROWS = 20000

SUFFIX = f"_nrows={NROWS}_generic"

# Word Embeddings Path
LOCAL_MODEL_PATH = "../gensim_embeddings"
GENSIM_WORD_EMBEDDINGS_PATH = "word2vec-google-news-300"
MODEL_FILENAME = GENSIM_WORD_EMBEDDINGS_PATH

AGGREGATION = "sum"

# Data to read from
POSITIVE_COMMENTS_CSV = "../../dataset/kaggle_reddit_comments/comments_positive.csv"
NEGATIVE_COMMENTS_CSV = "../../dataset/kaggle_reddit_comments/comments_negative.csv"
FILENAMES = [POSITIVE_COMMENTS_CSV, NEGATIVE_COMMENTS_CSV]


COLUMNS = ["text", "parent_text", "score", "parent_score"]

# Where to store models
SAVED_MODELS_FILEPATH = "saved_models"
LINEAR_REGRESSION_FILENAME = f"linear_regression{SUFFIX}.pkl"
SVR_FILENAME = f"support_vector_regressor{SUFFIX}.pkl"
MLP_FILENAME = f"nn_regressor{SUFFIX}.pkl"


def main():
    X, y = read_data(FILENAMES, transform_y=True)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

    print(y.describe())

    # Get embedding vectors
    w2v_model = get_model(MODEL_FILENAME)
    get_embed = lambda x: get_sentence_embedding(w2v_model, x, aggregation="avg")
    X_train['embeds'] = X_train['text'].apply(get_embed)
    X_test['embeds'] = X_test['text'].apply(get_embed)

    unpack_lambda = (lambda x: pd.Series(x))

    X_train_arr = X_train['embeds'].apply(unpack_lambda)
    X_test_arr = X_test['embeds'].apply(unpack_lambda)
    X_train_arr = X_train_arr.fillna(0)
    X_test_arr = X_test_arr.fillna(0)

    # Linear Regression
    print("training linear regressor...")
    linear_regression = models_utils.train_model(X_train_arr, y_train, model_type="linear_regression")
    LR_PATH = os.path.join(SAVED_MODELS_FILEPATH, LINEAR_REGRESSION_FILENAME)
    models_utils.save_model(linear_regression, LR_PATH, should_pickle=True)

    print('evaluating linear regressor...')
    score = models_utils.test_model(linear_regression, X_test_arr, y_test)
    print('score: ', score)

    # Support Vector Regression
    print("training support vector regressor...")
    support_vector_regression = models_utils.train_model(X_train_arr, y_train, model_type="support_vector_regression")
    SVR_PATH = os.path.join(SAVED_MODELS_FILEPATH, SVR_FILENAME)
    models_utils.save_model(support_vector_regression, SVR_PATH, should_pickle=True)

    print("evaluating support vector regressor...")
    score = models_utils.test_model(support_vector_regression, X_test_arr, y_test)
    print('score: ', score)

    # Neural Network
    print("training neural network...")
    mlp_regression = models_utils.train_model(X_train_arr, y_train, model_type="mlp_regression")
    MLP_PATH = os.path.join(SAVED_MODELS_FILEPATH, MLP_FILENAME)
    models_utils.save_model(mlp_regression, MLP_PATH, should_pickle=True)

    print("evaluating neural network...")
    score = models_utils.test_model(mlp_regression, X_test_arr, y_test)
    print('score: ', score)


def read_data(filenames, filter_scores=True, transform_y=False):
    df = pd.concat([
        pd.read_csv(filename, usecols=COLUMNS, na_filter=False, nrows=NROWS) for filename in filenames
    ], ignore_index=True)

    y = df['score']
    df.drop(columns='score', inplace=True)
    X = df

    # todo: filter scores

    # transform y
    if transform_y:
        y = transform_score(y, parent_scores=df['parent_score'])

    # todo: normalize y?

    return X, y


def transform_score(comment_scores, parent_scores):
    # todo: implement score transformation
    assert len(comment_scores) == len(parent_scores)
    for i, (comment_score, parent_score) in enumerate(zip(comment_scores, parent_scores)):
        if parent_score == 0:
            comment_scores[i] = 0
        else:
            comment_scores[i] = comment_scores[i] / parent_scores[i]
    return comment_scores


def safe_train_test_split(X, y, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=SEED)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()