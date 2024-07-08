from sklearn.model_selection import train_test_split
import pandas as pd


# Predicting child's height based on family data
def get_regression_data():
    df = pd.read_csv("GaltonFamilies.csv")
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    df = df.drop(columns=['family'])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Predicting lung cancer based on medical conditions
def get_classification_data():
    df = pd.read_csv("dataset.csv")
    df = pd.get_dummies(df, columns=df.columns.difference(["AGE"]), drop_first=True)
    df = df.astype(int) if df.dtypes.eq('bool').any() else df
    y = df["LUNG_CANCER_YES"].to_numpy()
    X = df.drop(columns=["LUNG_CANCER_YES"]).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
