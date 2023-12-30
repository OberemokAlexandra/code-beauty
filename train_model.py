import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


from config import MODEL_CONFIG, N_SPLITS, RANDOM_STATE, TEST_SIZE, MODEL_NAME


def split_data(password_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""
	Splits dataset into training and testing sets for machine learning modeling
	:param password_data: DataFrame to split
	:return: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training and testing sets
		for the X features and y target variable
	"""
	if 'password' in password_data.columns:
		password_data.drop('password', axis=1, inplace=True)
	X, X_test, y, y_test = train_test_split(password_data.iloc[:, 1:], password_data.iloc[:, 0],
											test_size=TEST_SIZE,
											random_state=RANDOM_STATE,
											stratify=password_data.iloc[:, 0])
	return X, X_test, y, y_test


def train_model(X: pd.DataFrame, y: pd.Series) -> None:
	"""
	Trains a logistic regression model on the given X features and y target variable using k-fold cross validation
	:param X: A pandas DataFrame containing features
	:param y: A pandas Series containing the y target variable
	:return: None
	"""
	logreg = LogisticRegression(**MODEL_CONFIG)
	cross_val = StratifiedKFold(n_splits=N_SPLITS)
	for train_index, test_index in cross_val.split(X, y):
		X_train, X_val = X.iloc[train_index], X.iloc[test_index]
		y_train, y_val = y.iloc[train_index], y.iloc[test_index]
		logreg.fit(X_train, y_train)
		y_pred = logreg.predict(X_val)
		print(f"{' f1 weighted score ':=^35}")
		print(f"{f1_score(y_pred, y_val, average='weighted'):.4f}")
		print(f"{' f1 macro score ':=^35}")
		print(f"{f1_score(y_pred, y_val, average='macro'):.4f}")
		report = classification_report(y_pred, y_val)
		print(f"{' classification report ':=^35}")
		print(f"{report}")
	pickle.dump(logreg, open(f"{MODEL_NAME}.pkl", "wb"))



