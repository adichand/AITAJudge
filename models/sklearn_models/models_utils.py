import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

MODEL_TYPES = ["linear_regression",
               "support_vector_regression",
               "mlp_regression"]


def train_model(X_train, y_train, model_type="mlp_regression"):
    # Declare model pipeline
    if model_type == "support_vector_regression":
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('regressor', SVR(kernel='rbf'))
        ])
    elif model_type == "mlp_regression":
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('regressor', MLPRegressor(hidden_layer_sizes=(256, 126), early_stopping=True))
        ])
    else: #model_type == "linear_regression":
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('regressor', LinearRegression())
        ])

    # Fit model pipeline
    print('fitting pipeline...')
    # print(X_train, y_train)

    pipeline.fit(X_train, y_train)

    return pipeline


def test_model(pipeline, X_test, y_test, scoring_method="default_regressor"):
    test_score = 0
    if scoring_method == "default_regressor":
        test_score = pipeline.score(X_test, y_test)

    return test_score


def save_model(pipeline, filename, should_pickle=True):
    if should_pickle:
        with open(filename, 'wb') as f:
            pickle.dump(pipeline, f)
