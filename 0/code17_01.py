from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


#数据导入 & 分割 & 标准化
digits = load_digits()
print(digits.data.shape)

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
print(y_train.shape, y_test.shape)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#训练模型
lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
y_predict = lsvc.predict(X_test)

score_lsvc = lsvc.score(X_test, y_test)
print('Accuarcy_lsvc:', score_lsvc)
print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))
