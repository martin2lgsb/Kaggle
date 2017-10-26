import pandas as pd


train = pd.read_csv('./Titanic/train.csv')
test = pd.read_csv('./Titanic/test.csv')

print(train.info())
print(test.info())
