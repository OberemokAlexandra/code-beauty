import pandas as pd

from config import INPUT_FLG, LABEL_VALUES, SOURCE_LINK
from predict import predict_passwords, predict_test_data
from preprocessing import check_data, get_data, get_features
from train_model import split_data, train_model


def prepare_dataframe() -> pd.DataFrame:
    """
    Prepares a pandas DataFrame by retrieving data from a specified source link,
    perform basic EDA, and adding relevant features from the data
    :return: pandas DataFrame, ready for modeling
    """
    data = get_data(link=SOURCE_LINK)
    check_data(data)
    return get_features(data)


def get_ml_model(data: pd.DataFrame) -> None:
    """
    Trains a machine learning model using the input data, splits the data into training and testing sets,
    and evaluates the model's performance on the testing set
    :param data: A pandas DataFrame containing the data to be used for training and testing the machine learning model
    :return:
    """
    x, x_test, y, y_test = split_data(data)
    print(f"{' train ML model ':=^35} \n")
    train_model(x, y)
    print(f"{' check metrics on test data ':=^35} \n")
    predict_test_data(x_test, y_test)


def get_password_strength(input_passwords: list) -> list:
    """
    Predicts strength of input list of passwords
    :param input_passwords: List of passwords
    :return: List of tuples like ('password_value', 'predicted_strength')
    """
    to_predict = [i.rstrip() for i in input_passwords if i.rstrip()]
    result = predict_passwords(to_predict)
    return list(zip(to_predict, result))


if __name__ == "__main__":
    password_data = prepare_dataframe()
    get_ml_model(password_data)
    if INPUT_FLG:
        passwords = input(f"Введите пароли через пробел: ").split()
    else:
        passwords = ["50BLOODYboiledcabb@ges", "i_am_groot", "weak"]
    predictions = pd.DataFrame(
        get_password_strength(passwords), columns=["password", "predicted_label"]
    )

    predictions["strength"] = predictions["predicted_label"].map(LABEL_VALUES)
    print(predictions)
