# Inspiration taken from https://www.kaggle.com/code/sunruslan/ai-group-jetbrains-internship

# Library imports
import pandas as pd

from sklearn.model_selection import train_test_split

from models.sklearn_models import models_utils

from models.sklearn_models.embed_sentences import get_model
from models.sklearn_models.embed_sentences import get_sentence_embedding

SEED = 42
BATCH_SIZE = 32
NROWS = 1000

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
LINEAR_REGRESSION_FILENAME = "linear_regression.pkl"
SVR_FILENAME = "support_vector_regressor.pkl"
MLP_FILENAME = "nn_regressor.pkl"

def main():
    X, y = read_data(FILENAMES)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

    # Get embedding vectors
    w2v_model = get_model(MODEL_FILENAME)
    get_embed = lambda x: get_sentence_embedding(w2v_model, x, aggregation="sum")
    X_train['embeds'] = X_train['text'].apply(get_embed)
    X_test['embeds'] = X_test['text'].apply(get_embed)

    unpack_lambda = (lambda x: pd.Series(x))

    X_train_arr = X_train['embeds'].apply(unpack_lambda)
    # print(len(X_train_arr.columns))

    # print(X_train_arr.describe())
    X_train_arr = X_train_arr.fillna(0)
    # print(X_train_arr.describe())
    # print(X_train['embeds'].describe())

    vector_size = w2v_model.vector_size
    train_size = len(X_train)
    test_size = len(X_test)

    # Linear Regression
    linear_regression = models_utils.train_model(X_train_arr, y_train, model_type="linear_regression")
    models_utils.save_model(linear_regression, LINEAR_REGRESSION_FILENAME, should_pickle=True)

    # Support Vector Regressor
    support_vector_regression = models_utils.train_model(X_train_arr, y_train, model_type="support_vector_regression")
    models_utils.save_model(support_vector_regression, SVR_FILENAME)

    # Neural Network
    mlp_regression = models_utils.train_model(X_train_arr, y_train, model_type="mlp_regression")
    models_utils.save_model(mlp_regression, MLP_FILENAME)


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
    for i, comment_score, parent_score in enumerate(zip(comment_scores, parent_scores)):
        if parent_score == 0:
            comment_score[i] = 0
        else:
            comment_score[i] = comment_score[i] / parent_score[i]
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