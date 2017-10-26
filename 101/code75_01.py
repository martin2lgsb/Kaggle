import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#数据导入 & 分割
train = pd.read_csv('./Breast-Cancer/breast-cancer-train.csv')
test = pd.read_csv('./Breast-Cancer/breast-cancer-test.csv')

X_train = np.float32(train[['Clump Thickness', 'Cell Size']].T)
y_train = np.float32(train['Type'].T)
X_test = np.float32(test[['Clump Thickness', 'Cell Size']].T)
y_test = np.float32(test['Type'].T)

#线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, X_train) + b

#优化目标
loss = tf.reduce_mean(tf.square(y - y_train))   #均方误差
optimizer = tf.train.GradientDescentOptimizer(0.01)    #梯度下降法，估参
train = optimizer.minimize(loss)    #最小二乘损失，优化

#训练模型
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(0, 1000):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(W), sess.run(b))

#作图
test_negative = test.loc[test['Type'] == 0][['Clump Thickness', 'Cell Size']]
test_positive = test.loc[test['Type'] == 1][['Clump Thickness', 'Cell Size']]

plt.scatter(test_negative['Clump Thickness'], test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(test_positive['Clump Thickness'], test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

lx = np.arange(0, 12)
ly = (0.5 - sess.run(b) - lx * sess.run(W)[0][0]) / sess.run(W)[0][1]

plt.plot(lx, ly, color = 'green')
plt.show()
