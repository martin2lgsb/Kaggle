import skflow
from sklearn import datasets, metrics, preprocessing, cross_validation
from sklearn.ensemble import RandomForestRegressor


#数据导入 & 分割 & 标准化
boston = datasets.load_boston()
X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=33)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#skflow-LinearRegressor
# tf_lr = skflow.TensorFlowLinearRegressor(steps=10000, learning_rate=0.01, batch_size=50)
# tf_lr.fit(X_train, y_train)
# tf_lr_y_predict = tf_lr.predict(X_test)
# print('mean_absolute_error:', metrics.mean_absolute_error(tf_lr_y_predict, y_test))
# print('mean_squared_error:', metrics.mean_squared_error(tf_lr_y_predict, y_test))
# print('r2_score:', metrics.r2_score(tf_lr_y_predict, y_test))

#skflow-DNNRegressor
# tf_dnn = skflow.TensorFlowDNNRegressor(hidden_units=[100, 40], steps=10000, learning_rate=0.01, batch_size=50)
# tf_dnn.fit(X_train, y_train)
# tf_dnn_y_predict = tf_dnn.predict(X_test)
# print('mean_absolute_error:', metrics.mean_absolute_error(tf_dnn_y_predict, y_test))
# print('mean_squared_error:', metrics.mean_squared_error(tf_dnn_y_predict, y_test))
# print('r2_score:', metrics.r2_score(tf_dnn_y_predict, y_test))

#sklearn-rfr
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)
print('mean_absolute_error:', metrics.mean_absolute_error(rfr_y_predict, y_test))
print('mean_squared_error:', metrics.mean_squared_error(rfr_y_predict, y_test))
print('r2_score:', metrics.r2_score(rfr_y_predict, y_test))
