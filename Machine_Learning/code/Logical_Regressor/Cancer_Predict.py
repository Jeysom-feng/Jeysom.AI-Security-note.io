import numpy as up
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("D:\\Desktop\\Python_file\\ML_Project\\Logical_Regressor\\breast_cancer_labeled.csv")  # 假设数据集为乳腺癌数据集
x_data = data.iloc[:, 2:30].values  # 特征数据
y_data = data.iloc[:, 31].values  # 标签数据
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

Scaler = StandardScaler()
x_train =Scaler.fit_transform(x_train)
x_test = Scaler.transform(x_test)

estimator =LogisticRegression()

# 训练模型
estimator.fit(x_train,y_train)

# 预测
y_pred = estimator.predict(x_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)