# -*- coding: utf-8 -*-
"""
Created on Tue May 09 10:57:24 2017

随机森林模型预测imdb电影评分
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# 加载数据集
df = pd.read_csv("movie_metadata.csv")
df =df.dropna()


# 数据预处理，将非数值型数据转换为数值型
from sklearn.preprocessing import LabelEncoder
columnsToEncode = list(df.select_dtypes(include=['category','object']))
le = LabelEncoder()
for feature in columnsToEncode:
    try:
        df[feature] = le.fit_transform(df[feature])
    except:
        print('Error encoding ' + feature)

X = df
y = X['imdb_score']
X = X.drop(['imdb_score'], axis=1)

# 数据分割，75%作为训练集，25%作为测试集
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X = scaler.fit_transform(X)
y = np.array(y).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


# 用随机深林模型来预测
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X_train, y_train)
clf_y_predict = clf.predict(X_test)
print(accuracy_score(y_test, clf_y_predict))
