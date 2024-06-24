from sklearn.model_selection import train_test_split
import pandas as pd


# Predicting child's height based on family data
def get_data():
    df = pd.read_csv("GaltonFamilies.csv")
    df['gender'] = df['gender'].replace({'male': 1, 'female': 0})
    df = df.drop(columns=['family'])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
