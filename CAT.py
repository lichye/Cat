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

print("Author: Leiqi Ye")
print("Contact: yeleiqi@umich.edu")

# 读取Excel文件
file_path = 'Data.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 创建LabelEncoder对象
le = LabelEncoder()

# 创建一个字典用于保存每列的LabelEncoder对象
label_encoders = {}

# 遍历所有列
for col in df.columns:
    # 如果该列的数据类型为object（可能是字符串）
    if df[col].dtype == 'object':

         # 将该列中的所有值都转化为字符串形式
        df[col] = df[col].apply(lambda x: str(x))
        
        # 对该列进行Label Encoding
        df[col] = le.fit_transform(df[col])

        # 保存该列的LabelEncoder对象
        label_encoders[col] = le

# with open('label_encoders.pkl', 'wb') as f:
#     pickle.dump(label_encoders, f)

df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: 1 if x > 0 else x)

#remove line with NaN
df_cleaned = df.dropna()

X = df_cleaned.iloc[:, :-1].values  # 特征，取所有行，除了最后一列的所有列
y = df_cleaned.iloc[:, -1].values   # 标签，取最后一列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)

# 初始化KNN分类器，设置k值为3
knn = KNeighborsClassifier(n_neighbors=10)
# 训练模型
knn.fit(X_train, y_train)
# 预测测试集
knn_train_pred = knn.predict(X_train)
knn_test_pred = knn.predict(X_test)
# 计算并打印准确率
knn_train_accuracy = accuracy_score(y_train,knn_train_pred)
knn_test_accuracy = accuracy_score(y_test, knn_test_pred)

knn_scores = cross_val_score(knn, X, y, cv=10)

print("finished KNN")



#朴素贝叶斯 (Naive Bayes)

Bay = GaussianNB()
Bay.fit(X_train, y_train)

Bay_train_pred = Bay.predict(X_train)
Bay_test_pred = Bay.predict(X_test)

accuracy_train_Bay = accuracy_score(y_train,Bay_train_pred)
accuracy_test_Bay = accuracy_score(y_test,Bay_test_pred)

Bay_scores = cross_val_score(Bay, X, y, cv=10)

print("finished Bayes")

#GBM
gbm = GradientBoostingClassifier(n_estimators=100)
gbm.fit(X_train, y_train)

gbm_train_pred = gbm.predict(X_train)
gbm_test_pred = gbm.predict(X_test)

accuracy_train_gbm  = accuracy_score(y_train,gbm_train_pred)
accuracy_test_gbm   = accuracy_score(y_test,gbm_test_pred)

gbm_scores = cross_val_score(gbm, X, y, cv=10)

print("finished GBM")

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


rf = RandomForestClassifier(n_estimators=100,max_depth=5)
rf.fit(X_train, y_train)

rf_train_pred = rf.predict(X_train)
rf_test_pred = rf.predict(X_test)

accuracy_train_rf = accuracy_score(y_train,rf_train_pred)
accuracy_test_rf = accuracy_score(y_test,rf_test_pred)

rf_scores = cross_val_score(rf, X, y, cv=10)

print("finished RF")

with open('result.txt','w') as f:
    sys.stdout = f 
    print(f'KNN accuracy rate on train:{knn_train_accuracy:.3f}')
    print(f'KNN accuracy rate on test: {knn_test_accuracy:.3f}')
    print(f'10-fold cross-validation result: {knn_scores}')
    print()

    print(f'Bay accuracy on train: {accuracy_train_Bay:.3f}')
    print(f'Bay accuracy on test: {accuracy_test_Bay:.3f}')
    print(f'10-fold cross-validation result: {Bay_scores}')
    print()

    print(f'GBM accuracy on train: {accuracy_train_gbm:.3f}')
    print(f'GBM accuracy on test: {accuracy_test_gbm:.3f}')
    print(f'10-fold cross-validation result: {gbm_scores}')
    print()

    print(f'MLP accuracy on train: {accuracy_train_mlp:.3f}')
    print(f'MLP accuracy on test: {accuracy_test_mlp:.3f}')
    print(f'10-fold cross-validation result: {mlp_scores}')
    print()

    print(f'dtc accuracy rate on train: {accuracy_train_dtc:.3f}')
    print(f'dtc accuracy rate on test: {accuracy_test_dtc:.3f}')
    print(f'10-fold cross-validation result: {dtc_scores}')
    print()

    print(f'kmeans accuracy rate on train: {accuracy_train_kmeans:.3f}')
    print(f'kmeans accuracy rate on test: {accuracy_test_kmeans:.3f}')
    print(f'10-fold cross-validation result: {kmeans_scores}')
    print()

    print(f'Rf accuracy rate on train: {accuracy_train_rf:.3f}')
    print(f'Rf accuracy rate on test: {accuracy_test_rf:.3f}')
    print(f'10-fold cross-validation result: {rf_scores}')
    print()

sys.stdout = sys.__stdout__