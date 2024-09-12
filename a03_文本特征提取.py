from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba
def count_demo():
    """
    文本特征抽取：CountVectorizer,统计每个样本中每个特征词出现个数
    :return:
    """
    data=["life is short,i like like python","life is too long,i dislike python"]
    # 1.实例化转换器类
    transfer=CountVectorizer(stop_words=['like'])# 去除停用词
    # 2.调用fit_transform
    data_new=transfer.fit_transform(data)
    print(f"data_new:{data_new.toarray()},{transfer.get_feature_names_out()}")# 文本特征抽取用不了sparse=False
    return None

def count_chinese_demo():
    """
    文本特征抽取：CountVectorizer,统计每个样本中每个特征词出现个数
    :return:
    """
    data=["我 爱 北京 天安门","天安门 上 太阳 升"]# 中文自己加空格，单字会被忽略
    # 1.实例化转换器类
    transfer=CountVectorizer()
    # 2.调用fit_transform
    data_new=transfer.fit_transform(data)
    print(f"data_new:{data_new.toarray()},{transfer.get_feature_names_out()}")# 文本特征抽取用不了sparse=False
    return None

def count_chinese_demo2():
    """
    中文文本特征抽取，自动分词
    :return:
    """
    data=["一种还是一种今天很残酷,明天更残酷,后天很美好,但绝对大部分是死在明天晚上,所以每个人不要放弃今天。",
         "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
          "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new=[]
    for sentence in data:
        data_new.append(cut_word(sentence))
    # 1.实例化转换器类
    transfer=CountVectorizer()
    # 2.调用fit_transform
    data_new1=transfer.fit_transform(data_new)
    print(f"data_new1:{data_new1.toarray()},{transfer.get_feature_names_out()}")

    return None

def cut_word(text):
    """
    进行中文分词
    :param text:
    :return:
    """
    text=" ".join(list(jieba.cut(text)))
    return text


def tfidf_demo():
    """
    用TF-IDF方法进行文本特征抽取
    :return:
    """
    data = ["一种还是一种今天很残酷,明天更残酷,后天很美好,但绝对大部分是死在明天晚上,所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new = []
    for sentence in data:
        data_new.append(cut_word(sentence))
    # 1.实例化转换器类
    transfer = TfidfVectorizer(stop_words=["一种","所以"])
    # 2.调用fit_transform
    data_new1 = transfer.fit_transform(data_new)
    print(f"data_new1:{data_new1.toarray()},{transfer.get_feature_names_out()}")
    return None
if __name__ == '__main__':
    # count_demo()
    # count_chinese_demo()
    # count_chinese_demo2()
    tfidf_demo()