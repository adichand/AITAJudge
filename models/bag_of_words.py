from models.load_dataset import get_dataframe
from models.load_dataset import get_train_test_split

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATASETS = ["../dataset/kaggle_reddit_comments/comments_positive.csv",
            "../dataset/kaggle_reddit_comments/comments_negative.csv"]
TEXT_COL = "text"
LABEL_COL = "score"

data_dfs = []
for dataset in DATASETS:
    data_df = get_dataframe(dataset, TEXT_COL, LABEL_COL, dataset_size=5000)
    data_dfs.append(data_df)

final_df = pd.concat(data_dfs)

X_train, X_test, y_train, y_test = get_train_test_split(final_df, TEXT_COL, LABEL_COL)

# Use CountVectorizer to create bag of words classifier
vectorizer = CountVectorizer(stop_words='english')
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train a neural network on bag of words
model = MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=1000)
model = model.fit(X_train_bow, y_train)

# Predict the scores for the testing set
y_pred = model.predict(X_test_bow)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

test_comment = "this is my test comment"
test_X = vectorizer.transform([test_comment])

prediction = model.predict(test_X)

print(prediction)