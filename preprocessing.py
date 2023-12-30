import re

import pandas as pd

from config import DATA_PATH


def get_data(link: str) -> pd.DataFrame:
    """
    Download data from link provided. If can't do so, takes a local copy
    :param link: A URL string pointing to the data source
    :return: A pandas DataFrame containing the downloaded data
    """
    try:
        input_data = pd.read_csv(link)
    except:
        print(f"Can't download data with link provided, get local copy")
        input_data = pd.read_csv(DATA_PATH)
    else:
        print(f"Data is loaded successfully")
    return input_data


def good_length_transformer(x: str) -> int:
    """
    Checks if password has a good length
    :param x: A password to be checked for its length
    :return: int: Returns 1 if the length of the string is greater than 7 characters, otherwise returns 0
    """
    return 1 if len(x) > 7 else 0


def is_upper_transformer(x: str) -> int:
    """
    Checks if password has any upper-case symbols
    :param x: A password to be checked
    :return: int: Returns 1 if the password contains upper-case symbols, otherwise returns 0
    """
    return 1 if any(char.isupper() for char in x) else 0


def is_digit_transformer(x: str) -> int:
    """
    Checks if password has any digits
    :param x: A password to be checked
    :return: int: Returns 1 if the password contains digits, otherwise returns 0
    """
    return 1 if any(char.isdigit() for char in x) else 0


def is_special_transformer(x: str) -> int:
    """
    Checks if password has any special symbols
    :param x: A password to be checked
    :return: int: Returns 1 if the password contains special symbols, otherwise returns 0
    """
    return 1 if bool(re.findall("[^.a-zA-Z\d]", x)) else 0


def check_data(input_data: pd.DataFrame) -> None:
    """
    Performs basic exploratory data analysis (EDA) on a input dataset
    :param input_data: A pandas DataFrame to be analyzed
    :return: None: This function does not return anything, but prints out various EDA metrics
    """
    print(f"{' First look ':=^35}")
    print(input_data.head(5))
    print(f"{' Input data shape ':=^35} \n"
          f"Length: {input_data.shape[0]:,} \n"
          f"Number of columns: {input_data.shape[1]:,}")
    print(f"{' Column types ':=^35}")
    print(input_data.dtypes)
    print(f"{' Null values count ':=^35}")
    print(input_data.isnull().sum(axis=0))
    is_duplicates = input_data['password'].duplicated().any()
    print(f"{' Duplicates check in passwords ':=^35}")
    if not is_duplicates:
        print(f"No duplicates")
    else:
        print(f"Duplicates in passwords are detected")
        duplicates = input_data.groupby('password')
        print(duplicates.filter(lambda x: len(x) > 1))
    print(f"{' Target distribution ':=^35}")
    print(input_data['strength'].value_counts())


def get_features(password_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates new features for a given password dataset by applying various transformers to the 'password' column
    :param password_data: A pandas DataFrame containing a 'password' column to be transformed.
    :return: Transformed pandas DataFrame.
    """
    password_data['is_good_length'] = password_data['password'].apply(
        good_length_transformer)
    password_data['is_upper'] = password_data['password'].apply(
        is_upper_transformer)
    password_data['is_digit'] = password_data['password'].apply(
        is_digit_transformer)
    password_data['is_special'] = password_data['password'].apply(
        is_special_transformer)
    return password_data
