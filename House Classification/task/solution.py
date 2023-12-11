import os
import random
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

    df = pd.read_csv("../Data/house_class.csv")

    X = df.loc[:,['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
    y = df.loc[:, 'Price']  # Use a Series for the target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=X['Zip_loc'].values, random_state=1)

    # One-hot encoding
    encoder_onehot = OneHotEncoder(drop='first')
    encoder_onehot.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

    # Transform training and test data with one-hot encoder
    X_train_onehot = pd.DataFrame(encoder_onehot.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(), index=X_train.index)
    X_test_onehot = pd.DataFrame(encoder_onehot.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(), index=X_test.index)

    # Join transformed features with original features
    X_train_final_onehot = X_train[['Area', 'Lon', 'Lat']].join(X_train_onehot)
    X_test_final_onehot = X_test[['Area', 'Lon', 'Lat']].join(X_test_onehot)

    # Convert column names to strings
    X_train_final_onehot.columns = X_train_final_onehot.columns.astype(str)
    X_test_final_onehot.columns = X_test_final_onehot.columns.astype(str)

    # Decision Tree Model with one-hot encoded data
    tree_onehot = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4, random_state=3)
    tree_onehot.fit(X_train_final_onehot, y_train)

    # Predict on the test data with one-hot encoded data
    y_pred_onehot = tree_onehot.predict(X_test_final_onehot)
    acc_onehot = accuracy_score(y_test, y_pred_onehot)

    # Ordinal encoding
    encoder_ordinal = OrdinalEncoder()
    encoder_ordinal.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

    # Transform training and test data with ordinal encoder
    X_train_ordinal = pd.DataFrame(encoder_ordinal.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]), index=X_train.index)
    X_test_ordinal = pd.DataFrame(encoder_ordinal.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]), index=X_test.index)

    # Join transformed features with original features
    X_train_final_ordinal = X_train[['Area', 'Lon', 'Lat']].join(X_train_ordinal)
    X_test_final_ordinal = X_test[['Area', 'Lon', 'Lat']].join(X_test_ordinal)

    # Convert column names to strings
    X_train_final_ordinal.columns = X_train_final_ordinal.columns.astype(str)
    X_test_final_ordinal.columns = X_test_final_ordinal.columns.astype(str)

    # Decision Tree Model with ordinal encoded data
    tree_ordinal = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4, random_state=3)
    tree_ordinal.fit(X_train_final_ordinal, y_train)

    # Predict on the test data with ordinal encoded data
    y_pred_ordinal = tree_ordinal.predict(X_test_final_ordinal)
    acc_ordinal = accuracy_score(y_test, y_pred_ordinal)

    # Target encoding
    encoder_target = TargetEncoder()
    encoder_target.fit(X_train, y_train)

    # Transform training and test data with target encoder
    X_train_target = encoder_target.transform(X_train)
    X_test_target = encoder_target.transform(X_test)

    # Decision Tree Model with target encoded data
    tree_target = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6,
                                         min_samples_split=4, random_state=3)
    tree_target.fit(X_train_target, y_train)

    # Predict on the test data with target encoded data
    y_pred_target = tree_target.predict(X_test_target)
    acc_target = accuracy_score(y_test, y_pred_target)

    # Model with OneHotEncoder
    y_pred_onehot = tree_onehot.predict(X_test_final_onehot)
    report_onehot = classification_report(y_test, y_pred_onehot, output_dict=True)
    f1_onehot = report_onehot['macro avg']['f1-score']

    # Model with OrdinalEncoder
    y_pred_ordinal = tree_ordinal.predict(X_test_final_ordinal)
    report_ordinal = classification_report(y_test, y_pred_ordinal, output_dict=True)
    f1_ordinal = report_ordinal['macro avg']['f1-score']

    # Model with TargetEncoder
    y_pred_target = tree_target.predict(X_test_target)
    report_target = classification_report(y_test, y_pred_target, output_dict=True)
    f1_target = report_target['macro avg']['f1-score']

    # Print the results
    print(f"OneHotEncoder:{round(f1_onehot, 2)}")
    print(f"OrdinalEncoder:{round(f1_ordinal, 2)}")
    print(f"TargetEncoder:{round(f1_target, 2)}")
