import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


#导入数据
df_train = pd.read_csv('./Breast-Cancer/breast-cancer-train.csv')
df_test = pd.read_csv('./Breast-Cancer/breast-cancer-test.csv')
print(df_train.shape, df_test.shape)

#特征值
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

#随机直线
# intercept = np.random.random([1])
# coef = np.random.random([2])
lx = np.arange(0, 12)
# ly = (- intercept - lx * coef[0]) / coef[1]
# plt.plot(lx, ly, c='yellow')

#训练数据
lr = LogisticRegression()
# lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
score = lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])
print(score)

intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]
plt.plot(lx, ly, c='blue')

#绘特征图
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()
