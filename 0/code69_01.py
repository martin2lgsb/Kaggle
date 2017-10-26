import nltk
from sklearn.feature_extraction.text import CountVectorizer

#数据导入 & 特征抽取 & 分割 & 正规化
sent1 = 'The cat is walking in the bedroom.'
sent2 = 'A dog was running across the kitchen.'

count_vec = CountVectorizer()
sentences = [sent1, sent2]
print(count_vec.fit_transform(sentences).toarray())
print(count_vec.get_feature_names())

tokens_1 = nltk.word_tokenize(sent1)
tokens_2 = nltk.word_tokenize(sent2)
print(tokens_1)
print(tokens_2)

vocab_1 = sorted(set(tokens_1))
vocab_2 = sorted(set(tokens_2))

# 寻找原始词根
stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(t) for t in tokens_1]
stem_2 = [stemmer.stem(t) for t in tokens_2]

# 标注
pos_tag_1 = nltk.tag.pos_tag(tokens_1)
pos_tag_2 = nltk.tag.pos_tag(tokens_2)
print(pos_tag_1)
print(pos_tag_2)
