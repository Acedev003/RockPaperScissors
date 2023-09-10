import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_PATH = "data/rps_1694359904.9626915.csv"

def load_data(path):
    df = pd.read_csv(path)
    print(df.head())    


if __name__ == '__main__':
    data = load_data(DATA_PATH)