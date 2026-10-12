import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

#导入数据
train = pd.read_csv('ML_Project\\data\\train.csv')
test = pd.read_csv('ML_Project\\data\\test.csv')
#训练集
x_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values
#测试集
#x_test = test.iloc[:, 1:].values
x_test = test.iloc[18, 1:].values
y_test = test.iloc[18, 0]
#交叉验证和网格搜索

#创建模型
pipe = Pipeline([
    ("scaler", MinMaxScaler(feature_range=(0, 1))),
    ("knn", KNeighborsClassifier())
])
#将模型交给网格搜索
param_grid = {'knn__n_neighbors': [3,5,7,9]}
estimator = sklearn.model_selection.GridSearchCV(pipe, param_grid, cv=5)
#训练模型
estimator.fit(x_train, y_train)
best_model = estimator.best_estimator_
#评估模型
y_pred = best_model.predict(x_test.reshape(1, -1))
print("Predicted:", y_pred, "Actual:", y_test)
#accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
#print("Best Parameters:", estimator.best_params_)
#print("Best Cross-Validation Score:", estimator.best_score_)
#保存模型
#joblib.dump(best_model, 'ML_Project\\model\\knn_model.pkl')