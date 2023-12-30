import pickle

import pandas as pd
from sklearn.metrics import classification_report, f1_score

from config import MODEL_NAME
from preprocessing import good_length_transformer, is_digit_transformer, is_special_transformer, is_upper_transformer


def load_model(model_name: str):
    """
    Loads a pre-trained logistic regression model from a pickle file
    :param model_name: The name of the pickle file containing the pre-trained model
    :return: The pre-trained logistic regression model
    """
    try:
        logreg = pickle.load(open(f"{model_name}.pkl", 'rb'))
    except:
        print(f"can't load model with path provided")
        raise
    else:
        return logreg


def predict_test_data(X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Calculates and prints the f1 weighted score, f1 macro score, and classification report
    for a logistic regression model on test data
    :param X_test: The test features
    :param y_test: The test labels
    :return: None This function does not return anything, but prints out various EDA metrics
    """
    logreg = load_model(MODEL_NAME)
    y_pred = logreg.predict(X_test)
    print(f"{' f1 weighted score ':=^35}")
    print(f"{f1_score(y_pred, y_test, average='weighted'):.4f}")
    print(f"{' f1 macro score ':=^35}")
    print(f"{f1_score(y_pred, y_test, average='macro'):.4f}")
    report = classification_report(y_pred, y_test)
    print(f"{' classification report ':=^35}")
    print(f"{report}")


def _prepare_chunk(chunk: str) -> tuple:
    """
    Prepares a single password for predict by calculating feature values
    :param chunk: The password chunk to be transformed
    :return: A tuple of transformed features for the chunk
    """
    if chunk:
        chunk_features = (good_length_transformer(chunk), is_upper_transformer(chunk), is_digit_transformer(chunk),
						  is_special_transformer(chunk))
        return chunk_features
    else:
        print('Empty password is detected')


def predict_passwords(passwords: list) -> list:
    """
    Predicts the strength of a list of passwords
    by transforming each password and applying a trained logistic regression model
    :param passwords: A list of passwords to be predicted
    :return: A list of predicted strengths for each password in the input
    """
    chunk_list = [_prepare_chunk(password) for password in passwords]
    cols = ["is_good_length", "is_upper", "is_digit", "is_special"]
    to_pred = pd.DataFrame(chunk_list, columns=cols)
    logreg = load_model(MODEL_NAME)
    results = logreg.predict(to_pred)
    return results
