
import configparser
import shap
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
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.metrics import f1_score
import configparser


def calculate_ppv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fp) if (tp + fp) > 0 else 0  # 避免除零错误

def calculate_sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0  # 避免除零错误

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def calculate_npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0  # 避免除零错误

def run_gbm():
    config = configparser.ConfigParser()
    config.read('config.ini')

    my_n_estimators = int(config.get('GBM Settings', 'n_estimators'))
    my_learning_rate = float(config.get('GBM Settings', 'learning_rate'))
    my_max_depth = int(config.get('GBM Settings', 'max_depth'))

    print("GBM Settings:")
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


    X = df_cleaned.iloc[:, :-1]  # features，take all columns except the last one
    y = df_cleaned.iloc[:, -1]   # Label，take the last column

    # dive the data into training and testing set
    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)


    #GBM
    gbm = GradientBoostingClassifier(n_estimators=my_n_estimators, learning_rate=my_learning_rate, max_depth=my_max_depth)
    gbm.fit(x_train, y_train)

    gbm_train_pred = gbm.predict(x_train)
    gbm_test_pred = gbm.predict(X_test)

    accuracy_train_gbm  = accuracy_score(y_train,gbm_train_pred)
    accuracy_test_gbm   = accuracy_score(y_test,gbm_test_pred)

    specificity_train_gbm = calculate_specificity(y_train, gbm_train_pred)
    specificity_test_gbm = calculate_specificity(y_test, gbm_test_pred)

    sensitivity_train_gbm = calculate_sensitivity(y_train, gbm_train_pred)
    sensitivity_test_gbm = calculate_sensitivity(y_test, gbm_test_pred)

    ppv_train_gbm = calculate_ppv(y_train, gbm_train_pred)
    ppv_test_gbm = calculate_ppv(y_test, gbm_test_pred)

    gbm_scores = cross_val_score(gbm, X, y, cv=10)
    
    pnv_train_gbm = calculate_npv(y_train, gbm_train_pred)
    pnv_test_gbm = calculate_npv(y_test, gbm_test_pred)

    f1_train_gbm = f1_score(y_train, gbm_train_pred)
    f1_test_gbm = f1_score(y_test, gbm_test_pred)

    # 创建 SHAP Explainer
    explainer = shap.Explainer(gbm, x_train)
    shap_values = explainer.shap_values(x_train)

    print(x_train.shape)
    print(shap_values.shape)

    plt.figure()
    shap.summary_plot(shap_values, x_train, show=False)  # 关闭自动显示

    plt.savefig("results//shap_GBM.png", dpi=300, bbox_inches='tight')  # 保存为高分辨率 PNG
    plt.close()


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
        print(f'GBM accuracy on train: {accuracy_train_gbm:.3f}')
        print(f'GBM accuracy on test: {accuracy_test_gbm:.3f}')
        print(f'GBM 10-fold cross-validation result: {gbm_scores.mean():.3f}')
        print(f'GBM specificity on train: {specificity_train_gbm:.3f}')
        print(f'GBM specificity on test: {specificity_test_gbm:.3f}')
        print(f'GBM sensitivity on train: {sensitivity_train_gbm:.3f}')
        print(f'GBM sensitivity on test: {sensitivity_test_gbm:.3f}')
        print(f'GBM PPV on train: {ppv_train_gbm:.3f}')
        print(f'GBM PPV on test: {ppv_test_gbm:.3f}')
        print(f'GBM NPV on train: {pnv_train_gbm:.3f}')
        print(f'GBM NPV on test: {pnv_test_gbm:.3f}')
        print(f'GBM F1 on train: {f1_train_gbm:.3f}')
        print(f'GBM F1 on test: {f1_test_gbm:.3f}')

    sys.stdout = sys.__stdout__

def run_rf():
    # read the configuration file

    config = configparser.ConfigParser()
    config.read('config.ini')

    my_n_estimators = int(config.get('RF Settings', 'n_estimators'))
    my_max_depth = int(config.get('RF Settings', 'max_depth'))
    my_min_samples_split = int(config.get('RF Settings', 'min_samples_split'))

    train_file_path = config.get('RF Settings', 'train_file_path')

    print("RF Settings:")
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


    X = df_cleaned.iloc[:, :-1] # features，take all columns except the last one
    y = df_cleaned.iloc[:, -1]   # Label，take the last column

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

    sensitivity_train_rf = calculate_sensitivity(y_train, rf_train_pred)
    sensitivity_test_rf = calculate_sensitivity(y_test, rf_test_pred)

    specificity_train_rf = calculate_specificity(y_train, rf_train_pred)
    specificity_test_rf = calculate_specificity(y_test, rf_test_pred)

    ppv_train_rf = calculate_ppv(y_train, rf_train_pred)
    ppv_test_rf = calculate_ppv(y_test, rf_test_pred)

    pnv_train_rf = calculate_npv(y_train, rf_train_pred)
    pnv_test_rf = calculate_npv(y_test, rf_test_pred)

    f1_train_rf = f1_score(y_train, rf_train_pred)
    f1_test_rf = f1_score(y_test, rf_test_pred)

    # 创建 SHAP Explainer
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(x_train)[...,1]

    print(x_train.shape)
    print(shap_values.shape)

    plt.figure()
    shap.summary_plot(shap_values, x_train, show=False)  # 关闭自动显示
    plt.savefig("results//shap_rf.png", dpi=300, bbox_inches='tight')
    plt.close()


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
        print(f'RF accuracy on train: {accuracy_tran_rf:.3f}')
        print(f'RF accuracy on test: {accuracy_test_rf:.3f}')
        print(f'RF 10-fold cross-validation result: {rf_scores.mean():.3f}')
        print(f'RF sensitivity on train: {sensitivity_train_rf:.3f}')
        print(f'RF sensitivity on test: {sensitivity_test_rf:.3f}')
        print(f'RF specificity on train: {specificity_train_rf:.3f}')
        print(f'RF specificity on test: {specificity_test_rf:.3f}')
        print(f'RF PPV on train: {ppv_train_rf:.3f}')
        print(f'RF PPV on test: {ppv_test_rf:.3f}')
        print(f'RF NPV on train: {pnv_train_rf:.3f}')
        print(f'RF NPV on test: {pnv_test_rf:.3f}')
        print(f'RF F1 on train: {f1_train_rf:.3f}')
        print(f'RF F1 on test: {f1_test_rf:.3f}')

    sys.stdout = sys.__stdout__

def run_KNN():
    config = configparser.ConfigParser()
    config.read('config.ini')
    train_file_path = config.get('Others Settings', 'train_file_path')

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


    X = df_cleaned.iloc[:, :-1]  # features，take all columns except the last one
    y = df_cleaned.iloc[:, -1]   # Label，take the last column

    # dive the data into training and testing set
    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)


    # initialize the Knn model
    knn = KNeighborsClassifier(n_neighbors=10)

    knn.fit(x_train, y_train)

    knn_train_pred = knn.predict(x_train)
    knn_test_pred = knn.predict(X_test)

    knn_train_accuracy = accuracy_score(y_train,knn_train_pred)
    knn_test_accuracy = accuracy_score(y_test, knn_test_pred)

    knn_scores = cross_val_score(knn, X, y, cv=10)

    knn_sensitive_train = calculate_sensitivity(y_train, knn_train_pred)
    knn_sensitive_test = calculate_sensitivity(y_test, knn_test_pred)

    knn_specificity_train = calculate_specificity(y_train, knn_train_pred)
    knn_specificity_test = calculate_specificity(y_test, knn_test_pred)

    knn_ppv_train = calculate_ppv(y_train, knn_train_pred)
    knn_ppv_test = calculate_ppv(y_test, knn_test_pred)

    knn_npv_train = calculate_npv(y_train, knn_train_pred)
    knn_npv_test = calculate_npv(y_test, knn_test_pred)

    knn_f1_train = f1_score(y_train, knn_train_pred)
    knn_f1_test = f1_score(y_test, knn_test_pred)
    
    print("finished KNN")

    with open('results//knn_result.txt','w') as f:
        sys.stdout = f 
        print(f'KNN accuracy rate on train:{knn_train_accuracy:.3f}')
        print(f'KNN accuracy rate on test: {knn_test_accuracy:.3f}')
        print(f'KNN 10-fold cross-validation result means: {knn_scores.mean():.3f}')
        print(f'KNN sensitivity on train: {knn_sensitive_train:.3f}')
        print(f'KNN sensitivity on test: {knn_sensitive_test:.3f}')
        print(f'KNN specificity on train: {knn_specificity_train:.3f}')
        print(f'KNN specificity on test: {knn_specificity_test:.3f}')   
        print(f'KNN PPV on train: {knn_ppv_train:.3f}')
        print(f'KNN PPV on test: {knn_ppv_test:.3f}')
        print(f'KNN NPV on train: {knn_npv_train:.3f}')
        print(f'KNN NPV on test: {knn_npv_test:.3f}')
        print(f'KNN F1 on train: {knn_f1_train:.3f}')
        print(f'KNN F1 on test: {knn_f1_test:.3f}')
        print(f"KNN does not have SHAP value")
        print()

    sys.stdout = sys.__stdout__

def run_Bayes():
    # Read the configuration file
    config = configparser.ConfigParser()

    config.read('config.ini')

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

    X = df_cleaned.iloc[:, :-1]  # features
    y = df_cleaned.iloc[:, -1]   # Labels

    # divide the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)

    
    #Naive Bayes

    Bay = GaussianNB()
    Bay.fit(X_train, y_train)

    Bay_train_pred = Bay.predict(X_train)
    Bay_test_pred = Bay.predict(X_test)

    accuracy_train_Bay = accuracy_score(y_train,Bay_train_pred)
    accuracy_test_Bay = accuracy_score(y_test,Bay_test_pred)

    Bay_scores = cross_val_score(Bay, X, y, cv=10)

    Bay_train_sensitive = calculate_sensitivity(y_train, Bay_train_pred)
    Bay_test_sensitive = calculate_sensitivity(y_test, Bay_test_pred)

    Bay_train_specificity = calculate_specificity(y_train, Bay_train_pred)
    Bay_test_specificity = calculate_specificity(y_test, Bay_test_pred)

    Bay_train_ppv = calculate_ppv(y_train, Bay_train_pred)
    Bay_test_ppv = calculate_ppv(y_test, Bay_test_pred)

    Bay_train_npv = calculate_npv(y_train, Bay_train_pred)
    Bay_test_npv = calculate_npv(y_test, Bay_test_pred)

    Bay_train_f1 = f1_score(y_train, Bay_train_pred)
    Bay_test_f1 = f1_score(y_test, Bay_test_pred)
    
    print("finished Bayes")

    # sample_data = X_train.sample(n=50, random_state=42)  # 取 50 个样本作为背景
    # explainer = shap.KernelExplainer(Bay.predict_proba, sample_data)

    # shap_values = explainer.shap_values(X_train.iloc[:100])

    # plt.figure()
    # shap.summary_plot(shap_values, X_train.iloc[:100], show=False)  # 类别 

    # plt.savefig("results//shap_Bay.png", dpi=300, bbox_inches='tight')  # 保存为高分辨率 PNG
    # plt.close()

    with open('results//bayes_result.txt','w') as f:
        sys.stdout = f 
        print(f'Bay accuracy on train: {accuracy_train_Bay:.3f}')
        print(f'Bay accuracy on test: {accuracy_test_Bay:.3f}')
        print(f'Bay 10-fold cross-validation result: {Bay_scores.mean():.3f}')
        print(f'Bay sensitivity on train: {Bay_train_sensitive:.3f}')
        print(f'Bay sensitivity on test: {Bay_test_sensitive:.3f}')
        print(f'Bay specificity on train: {Bay_train_specificity:.3f}')
        print(f'Bay specificity on test: {Bay_test_specificity:.3f}')
        print(f'Bay PPV on train: {Bay_train_ppv:.3f}')
        print(f'Bay PPV on test: {Bay_test_ppv:.3f}')
        print(f'Bay NPV on train: {Bay_train_npv:.3f}')
        print(f'Bay NPV on test: {Bay_test_npv:.3f}')
        print(f'Bay F1 on train: {Bay_train_f1:.3f}')
        print(f'Bay F1 on test: {Bay_test_f1:.3f}')
        print()
    sys.stdout = sys.__stdout__

def run_mlp():
    # Read the configuration file
    config = configparser.ConfigParser()

    config.read('config.ini')

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

    X = df_cleaned.iloc[:, :-1]  # features
    y = df_cleaned.iloc[:, -1]   # Labels

    # divide the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)

     #MLP
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    mlp.fit(X_train, y_train)

    mlp_train_pred = mlp.predict(X_train)
    mlp_test_pred = mlp.predict(X_test)

    accuracy_train_mlp  = accuracy_score(y_train,mlp_train_pred)
    accuracy_test_mlp   = accuracy_score(y_test,mlp_test_pred)

    mlp_scores = cross_val_score(mlp, X, y, cv=10)

    mlp_train_sensitive = calculate_sensitivity(y_train, mlp_train_pred)
    mlp_test_sensitive = calculate_sensitivity(y_test, mlp_test_pred)

    mlp_train_specificity = calculate_specificity(y_train, mlp_train_pred)
    mlp_test_specificity = calculate_specificity(y_test, mlp_test_pred)

    mlp_train_ppv = calculate_ppv(y_train, mlp_train_pred)
    mlp_test_ppv = calculate_ppv(y_test, mlp_test_pred)

    mlp_train_npv = calculate_npv(y_train, mlp_train_pred)
    mlp_test_npv = calculate_npv(y_test, mlp_test_pred)

    mlp_train_f1 = f1_score(y_train, mlp_train_pred)
    mlp_test_f1 = f1_score(y_test, mlp_test_pred)

    print("finished MLP")

    with open('results//mlp_result.txt','w') as f:
        sys.stdout = f 
        print(f'MLP accuracy on train: {accuracy_train_mlp:.3f}')
        print(f'MLP accuracy on test: {accuracy_test_mlp:.3f}')
        print(f'MLP 10-fold cross-validation result: {mlp_scores.mean():.3f}')
        print(f'MLP sensitivity on train: {mlp_train_sensitive:.3f}')
        print(f'MLP sensitivity on test: {mlp_test_sensitive:.3f}')
        print(f'MLP specificity on train: {mlp_train_specificity:.3f}')
        print(f'MLP specificity on test: {mlp_test_specificity:.3f}')
        print(f'MLP PPV on train: {mlp_train_ppv:.3f}')
        print(f'MLP PPV on test: {mlp_test_ppv:.3f}')
        print(f'MLP NPV on train: {mlp_train_npv:.3f}')
        print(f'MLP NPV on test: {mlp_test_npv:.3f}')
        print(f'MLP F1 on train: {mlp_train_f1:.3f}')
        print(f'MLP F1 on test: {mlp_test_f1:.3f}')
        print()
    sys.stdout = sys.__stdout__

def run_dtc():
     # Read the configuration file
    config = configparser.ConfigParser()

    config.read('config.ini')

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

    X = df_cleaned.iloc[:, :-1]  # features
    y = df_cleaned.iloc[:, -1]   # Labels

    # divide the dataset into training and testing sets
    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)

    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)

    dtc_train_pred = dtc.predict(x_train)
    dtc_test_pred = dtc.predict(X_test)

    accuracy_train_dtc  = accuracy_score(y_train,dtc_train_pred)
    accuracy_test_dtc   = accuracy_score(y_test,dtc_test_pred)

    dtc_scores = cross_val_score(dtc, X, y, cv=10)

    dtc_train_sensitive = calculate_sensitivity(y_train, dtc_train_pred)
    dtc_test_sensitive = calculate_sensitivity(y_test, dtc_test_pred)

    dtc_train_specificity = calculate_specificity(y_train, dtc_train_pred)
    dtc_test_specificity = calculate_specificity(y_test, dtc_test_pred)

    dtc_train_ppv = calculate_ppv(y_train, dtc_train_pred)
    dtc_test_ppv = calculate_ppv(y_test, dtc_test_pred)

    dtc_train_npv = calculate_npv(y_train, dtc_train_pred)
    dtc_test_npv = calculate_npv(y_test, dtc_test_pred)

    dtc_train_f1 = f1_score(y_train, dtc_train_pred)
    dtc_test_f1 = f1_score(y_test, dtc_test_pred)
    
    print("finished dtc")


    with open('results//dtc_result.txt','w') as f:
        sys.stdout = f 
        print(f'DTC accuracy rate on train: {accuracy_train_dtc:.3f}')
        print(f'DTC accuracy rate on test: {accuracy_test_dtc:.3f}')
        print(f'DTC 10-fold cross-validation result: {dtc_scores.mean():.3f}')
        print(f'DTC sensitivity on train: {dtc_train_sensitive:.3f}')
        print(f'DTC sensitivity on test: {dtc_test_sensitive:.3f}')
        print(f'DTC specificity on train: {dtc_train_specificity:.3f}')
        print(f'DTC specificity on test: {dtc_test_specificity:.3f}')
        print(f'DTC PPV on train: {dtc_train_ppv:.3f}')
        print(f'DTC PPV on test: {dtc_test_ppv:.3f}')
        print(f'DTC NPV on train: {dtc_train_npv:.3f}')
        print(f'DTC NPV on test: {dtc_test_npv:.3f}')
        print(f'DTC F1 on train: {dtc_train_f1:.3f}')
        print(f'DTC F1 on test: {dtc_test_f1:.3f}')
        print()
    sys.stdout = sys.__stdout__

def run_Kmeans():
     # Read the configuration file
    config = configparser.ConfigParser()

    config.read('config.ini')

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

    X = df_cleaned.iloc[:, :-1]  # features
    y = df_cleaned.iloc[:, -1]   # Labels

    # divide the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)

     #Kmeans

    kmeans = KMeans(n_clusters=3,n_init=10)
    kmeans.fit(X_train)

    kmeans_train_pred = kmeans.predict(X_train)
    kmeans_test_pred = kmeans.predict(X_test)

    accuracy_train_kmeans = accuracy_score(y_train,kmeans_train_pred)
    accuracy_test_kmeans   = accuracy_score(y_test,kmeans_test_pred)

    kmeans_scores = cross_val_score(kmeans, X, y, cv=10)

    # kmeans_train_f1 = f1_score(y_train, kmeans_train_pred)
    # kmeans_test_f1 = f1_score(y_test, kmeans_test_pred)
    print("finished Kmeans")

    with open('results//kmeans_result.txt','w') as f:
        sys.stdout = f 
        print(f'Kmeans accuracy rate on train: {accuracy_train_kmeans:.3f}')
        print(f'Kmeans accuracy rate on test: {accuracy_test_kmeans:.3f}')
        print(f'Kmeans 10-fold cross-validation result: {kmeans_scores.mean():.3f}')
        # print(f'Kmeans F1 on train: {kmeans_train_f1:.3f}')
        # print(f'Kmeans F1 on test: {kmeans_test_f1:.3f}')
        print()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
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

    run_Bayes()
    run_KNN()
    run_mlp()
    run_dtc()
    run_Kmeans()
    
    print("finish running all the models")
    input("Press Enter to quit...")

