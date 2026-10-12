import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,cross_val_score

# 读取数据集
data =pd.read_csv("ML_Project/CART_Tree/titanic_train.csv")
#print(data.info())
#数据处理
x_train = data[["Pclass","Sex","Age"]].copy()
y_train = data["Survived"].copy()
#print(x_train.head(1))
#print(y_train.head(1))
x_train["Age"]=x_train["Age"].fillna(x_train["Age"].mean())
x_train=pd.get_dummies(x_train,columns=["Sex"],drop_first=True)
# print(x_train.info())
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2,random_state=42)
#创建随机森林模型
estimator =RandomForestClassifier()

param ={"n_estimators":[30,60,90,110],"max_depth":[2,3,5]}
gs_estimator =GridSearchCV(estimator,param,cv=2)
gs_estimator.fit(x_train,y_train)
print("得分：", gs_estimator.best_score_)
print("最优参数:",gs_estimator.best_params_)