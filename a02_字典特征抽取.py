from sklearn.feature_extraction import DictVectorizer
def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data=[{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}]
    # 1.实例化一个转换器类
    transfer=DictVectorizer(sparse=False)
    # 2.调用fit_transform(),传字典列表，默认返回稀疏矩阵，可改
    data_new=transfer.fit_transform(data)
    print(f"data_new:{data_new}")# 输出one-hot编码
    print(f"特征名：{transfer.get_feature_names_out()}")
    return None

if __name__ == '__main__':
    # 字典特征提取
    dict_demo()