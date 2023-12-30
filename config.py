SOURCE_LINK = "https://d93017f8-0573-411d-886e-48d58ad98a95.selcdn.net/task/ds/passwords.csv"
RANDOM_STATE = 42

TEST_SIZE = 0.2

N_SPLITS = 5

MODEL_CONFIG = {
	"multi_class": "multinomial",
	"solver": "lbfgs",
	"penalty": "l2",
	"class_weight": "balanced",
	"C": 1.0,
	"max_iter": 100,
}

LABEL_VALUES = {0: "weak", 1: "medium", 2: "strong"}
MODEL_NAME = "log_reg"
DATA_PATH = "passwords.csv"
INPUT_FLG = False
