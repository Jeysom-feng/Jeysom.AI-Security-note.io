import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error


# 读取数据（波士顿房价，GitHub 镜像已下载为本地 CSV）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_df = pd.read_csv(os.path.join(BASE_DIR, "BostonHousing.csv"))
data = raw_df.iloc[:, :-1].values  # 前13列为特征
target = raw_df["medv"].values  # 最后一列为目标房价

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 训练模型
estimator = LinearRegression()##改为SGDR则变为随机梯度下降法
estimator.fit(x_train, y_train)
print("Linear Regression Coefficients:", estimator.coef_)
print("Linear Regression Intercept:", estimator.intercept_)

# 模型预测
y_pred = estimator.predict(x_test)
print("Predicted Values:", y_pred)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
rmse = root_mean_squared_error(y_test, y_pred)
print("Root Mean Squared Error:", rmse)
