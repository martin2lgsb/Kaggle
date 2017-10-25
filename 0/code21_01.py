from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


#数据导入 & 分割 & 标准化
news = fetch_20newsgroups(subset='all')
print(len(news.data))
print(news.data[0])

X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
print(y_train.shape, y_test.shape)

vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

#训练模型
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_predict = mnb.predict(X_test)

score_mnb = mnb.score(X_test, y_test)
print('Accuarcy_mnb:', score_mnb)
print(classification_report(y_test, y_predict, target_names=news.target_names))
