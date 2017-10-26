import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report


#数据导入 & 清理 & 分割 & 标准化
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=column_names)
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')
print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)
print(y_train.value_counts())
print(y_test.value_counts())

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#训练模型
lr = LogisticRegression()
sgdc = SGDClassifier()

lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)

score_lr = lr.score(X_test, y_test)
score_sgdc = sgdc.score(X_test, y_test)
target_names = ['Benign', 'Malignant']
print('Accuarcy_lr:', score_lr)
print(classification_report(y_test, lr_y_predict, target_names=target_names))
print('Accuarcy_sgdc:', score_sgdc)
print(classification_report(y_test, sgdc_y_predict, target_names=target_names))
