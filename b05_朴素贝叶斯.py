# 朴素：特征与特征之间相互独立
# 朴素贝叶斯用于分类
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
def nb_news():
    """
    用朴素贝叶斯分类新闻
    :return:
    """
    news=fetch_20newsgroups(subset='all')
    x_train,x_test,y_train,y_test=train_test_split(news.data,news.target,)
    transfer=TfidfVectorizer()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    estimator=MultinomialNB()
    estimator.fit(x_train,y_train)

    y_predict=estimator.predict(x_test)
    print(y_predict)
    print(y_test==y_predict)
    score=estimator.score(x_test,y_test)
    print(score)
if __name__ == '__main__':
    nb_news()