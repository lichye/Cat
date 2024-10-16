
import configparser
import subprocess
import os
import pandas as pd
import time
import pickle
import sys
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import configparser



def run_gbm():
    config = configparser.ConfigParser()
    config.read('config.ini')
    print_train_result = config.get('GBM Settings', 'print_train_result') 
    print_test_result = config.get('GBM Settings', 'print_test_result')
    print_fold_result = config.get('GBM Settings', 'print_fold_result')
    my_n_estimators = int(config.get('GBM Settings', 'n_estimators'))
    my_learning_rate = float(config.get('GBM Settings', 'learning_rate'))
    my_max_depth = int(config.get('GBM Settings', 'max_depth'))

    print("GBM Settings:")
    print(f"print_train_result: {print_train_result}")
    print(f"print_test_result: {print_test_result}")
    print(f"print_fold_result: {print_fold_result}")
    print(f"n_estimators: {my_n_estimators}")
    print(f"learning_rate: {my_learning_rate}")
    print(f"max_depth: {my_max_depth}")


    # Read Excel file
    file_path = 'Data.xlsx'  # set the path of the Excel file
    df = pd.read_excel(file_path)

    # create a LabelEncoder object
    le = LabelEncoder()

    label_encoders = {}

    # Loop over all columns in the DataFrame
    for col in df.columns:
        # Check if the column is not numeric
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: str(x))
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Save the label encoders to a file
    # with open('label_encoders.pkl', 'wb') as f:
    #     pickle.dump(label_encoders, f)

    # Convert the last column to binary
    df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: 1 if x > 0 else x)

    #remove line with NaN
    df_cleaned = df.dropna()


    X = df_cleaned.iloc[:, :-1].values  # features，take all columns except the last one
    y = df_cleaned.iloc[:, -1].values   # Label，take the last column

    # dive the data into training and testing set
    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)


    #GBM
    gbm = GradientBoostingClassifier(n_estimators=my_n_estimators, learning_rate=my_learning_rate, max_depth=my_max_depth)
    gbm.fit(x_train, y_train)

    gbm_train_pred = gbm.predict(x_train)
    gbm_test_pred = gbm.predict(X_test)

    accuracy_train_gbm  = accuracy_score(y_train,gbm_train_pred)
    accuracy_test_gbm   = accuracy_score(y_test,gbm_test_pred)

    gbm_scores = cross_val_score(gbm, X, y, cv=10)

    print("finished GBM")

    feature_names = list(df_cleaned.columns)

    # print("Feature Importance in GBM/RF")
    # print(feature_names)
    importances = gbm.feature_importances_
    feature_names = feature_names[:len(importances)]

    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance in GBM")
    plt.tight_layout()
    plt.savefig("results//GBM_feature_importance.png")  # 保存为 PNG 格式


    with open('results//gbm_result.txt','w') as f:
        sys.stdout = f 
        if(print_train_result):
            print(f'GBM accuracy on train: {accuracy_train_gbm:.3f}')
            print()

        if(print_test_result):
            print(f'GBM accuracy on test: {accuracy_test_gbm:.3f}')
            print()
        
        if(print_fold_result):
            print(f'GBM 10-fold cross-validation result: {gbm_scores.mean():.3f}')
            print()
    sys.stdout = sys.__stdout__



def run_rf():
    # read the configuration file

    config = configparser.ConfigParser()
    config.read('config.ini')

    print_train_result = config.get('RF Settings', 'print_train_result')
    print_test_result = config.get('RF Settings', 'print_test_result')
    print_fold_result = config.get('RF Settings', 'print_fold_result')
    my_n_estimators = int(config.get('RF Settings', 'n_estimators'))
    my_max_depth = int(config.get('RF Settings', 'max_depth'))
    my_min_samples_split = int(config.get('RF Settings', 'min_samples_split'))

    train_file_path = config.get('RF Settings', 'train_file_path')

    print("RF Settings:")
    print(f"print_train_result: {print_train_result}")
    print(f"print_test_result: {print_test_result}")
    print(f"print_fold_result: {print_fold_result}")
    print(f"n_estimators: {my_n_estimators}")
    print(f"max_depth: {my_max_depth}")
    print(f"min_samples_split: {my_min_samples_split}")

    df = pd.read_excel(train_file_path)
    # create a LabelEncoder object
    le = LabelEncoder()

    label_encoders = {}

    # Loop over all columns in the DataFrame
    for col in df.columns:
        # Check if the column is not numeric
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: str(x))
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Save the label encoders to a file
    # with open('label_encoders.pkl', 'wb') as f:
    #     pickle.dump(label_encoders, f)

    # Convert the last column to binary
    df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: 1 if x > 0 else x)

    #remove line with NaN
    df_cleaned = df.dropna()


    X = df_cleaned.iloc[:, :-1].values  # features，take all columns except the last one
    y = df_cleaned.iloc[:, -1].values   # Label，take the last column

    # dive the data into training and testing set
    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)


    #RF
    rf = RandomForestClassifier(n_estimators=my_n_estimators, max_depth=my_max_depth, min_samples_split=my_min_samples_split)
    rf.fit(x_train, y_train)

    rf_train_pred = rf.predict(x_train)
    rf_test_pred = rf.predict(X_test) 

    accuracy_tran_rf = accuracy_score(y_train, rf_train_pred)
    accuracy_test_rf = accuracy_score(y_test, rf_test_pred)

    rf_scores = cross_val_score(rf, X, y, cv=10)


    print("Finished RF")

    feature_names = list(df_cleaned.columns)

    importances =   rf.feature_importances_
    feature_names = feature_names[:len(importances)]

    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance in RF")
    plt.tight_layout()
    plt.savefig("results//rf_feature_importance.png")


    with open('results//rf_result.txt','w') as f:
        sys.stdout = f 
        if(print_train_result):
            print(f'RF accuracy on train: {accuracy_tran_rf:.3f}')
            print()

        if(print_test_result):
            print(f'RF accuracy on test: {accuracy_test_rf:.3f}')
            print()
        
        if(print_fold_result):
            print(f'RF 10-fold cross-validation result: {rf_scores.mean():.3f}')
            print()
    sys.stdout = sys.__stdout__


def run_other():
        
    # Read the configuration file
    config = configparser.ConfigParser()

    config.read('config.ini')


    print_train_result = config.get('Others Settings', 'print_train_result')
    print_test_result = config.get('Others Settings', 'print_test_result')
    print_fold_result = config.get('Others Settings', 'print_fold_result')
    train_file_path = config.get('Others Settings', 'train_file_path')

    df = pd.read_excel(train_file_path)

    le = LabelEncoder()

    label_encoders = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: str(x))
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # with open('label_encoders.pkl', 'wb') as f:
    #     pickle.dump(label_encoders, f)

    df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: 1 if x > 0 else x)

    #remove line with NaN
    df_cleaned = df.dropna()

    X = df_cleaned.iloc[:, :-1].values  # features
    y = df_cleaned.iloc[:, -1].values   # Labels

    # divide the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)

    # initialize the Knn model
    knn = KNeighborsClassifier(n_neighbors=10)

    knn.fit(X_train, y_train)

    knn_train_pred = knn.predict(X_train)
    knn_test_pred = knn.predict(X_test)

    knn_train_accuracy = accuracy_score(y_train,knn_train_pred)
    knn_test_accuracy = accuracy_score(y_test, knn_test_pred)

    knn_scores = cross_val_score(knn, X, y, cv=10)

    print("finished KNN")

    #Naive Bayes

    Bay = GaussianNB()
    Bay.fit(X_train, y_train)

    Bay_train_pred = Bay.predict(X_train)
    Bay_test_pred = Bay.predict(X_test)

    accuracy_train_Bay = accuracy_score(y_train,Bay_train_pred)
    accuracy_test_Bay = accuracy_score(y_test,Bay_test_pred)

    Bay_scores = cross_val_score(Bay, X, y, cv=10)

    print("finished Bayes")

    #MLP
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    mlp.fit(X_train, y_train)

    mlp_train_pred = mlp.predict(X_train)
    mlp_test_pred = mlp.predict(X_test)

    accuracy_train_mlp  = accuracy_score(y_train,mlp_train_pred)
    accuracy_test_mlp   = accuracy_score(y_test,mlp_test_pred)

    mlp_scores = cross_val_score(mlp, X, y, cv=10)

    print("finished MLP")


    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)

    dtc_train_pred = dtc.predict(X_train)
    dtc_test_pred = dtc.predict(X_test)

    accuracy_train_dtc  = accuracy_score(y_train,dtc_train_pred)
    accuracy_test_dtc   = accuracy_score(y_test,dtc_test_pred)

    dtc_scores = cross_val_score(dtc, X, y, cv=10)

    print("finished dtc")

    #Kmeans

    kmeans = KMeans(n_clusters=3,n_init=10)
    kmeans.fit(X_train)

    kmeans_train_pred = kmeans.predict(X_train)
    kmeans_test_pred = kmeans.predict(X_test)

    accuracy_train_kmeans = accuracy_score(y_train,kmeans_train_pred)
    accuracy_test_kmeans   = accuracy_score(y_test,kmeans_test_pred)

    kmeans_scores = cross_val_score(kmeans, X, y, cv=10)

    print("finished Kmeans")


    with open('results//result.txt','w') as f:
        sys.stdout = f 
        if(print_train_result):
            print(f'KNN accuracy rate on train:{knn_train_accuracy:.3f}')
            print(f'Bay accuracy on train: {accuracy_train_Bay:.3f}')
            print(f'MLP accuracy on train: {accuracy_train_mlp:.3f}')
            print(f'dtc accuracy rate on train: {accuracy_train_dtc:.3f}')
            print(f'kmeans accuracy rate on train: {accuracy_train_kmeans:.3f}')
            print()

        if(print_test_result):
            print(f'KNN accuracy rate on test: {knn_test_accuracy:.3f}')
            print(f'Bay accuracy on test: {accuracy_test_Bay:.3f}')
            print(f'MLP accuracy on test: {accuracy_test_mlp:.3f}')
            print(f'dtc accuracy rate on test: {accuracy_test_dtc:.3f}')
            print(f'kmeans accuracy rate on test: {accuracy_test_kmeans:.3f}')
            print()
        
        if(print_fold_result):
            print(f'KNN 10-fold cross-validation result means: {knn_scores.mean():.3f}')
            print(f'Bay 10-fold cross-validation result: {Bay_scores.mean():.3f}')
            print(f'MLP 10-fold cross-validation result: {mlp_scores.mean():.3f}')
            print(f'DTC 10-fold cross-validation result: {dtc_scores.mean():.3f}')
            print(f'KMEANS 10-fold cross-validation result: {kmeans_scores.mean():.3f}')
            print()
    sys.stdout = sys.__stdout__




# Main function

print()
print("Author: Leiqi Ye")
print("Contact: yeleiqi@umich.edu")
print()
print()

folder_name = "results"

try:
    os.mkdir(folder_name)
except:
    print("Results Folder already exists")

config = configparser.ConfigParser()

config.read('config.ini')

# selection the GBM settings
gbm_control = config.get('GBM Settings', 'gbm_run') 

rf_control = config.get('RF Settings', 'rf_run')

other_control = config.get('Others Settings', 'others_run')

if(gbm_control):
    print("Start to run GBM")
    run_gbm()
    print()

if(rf_control):
    print("Start to run RF")
    run_rf()
    print()

if(other_control):
    print("Start to run Others")
    run_other()



print("finish running all the models")
input("Press Enter to quit...")

