from sklearn.decomposition import PCA
def pca_demo():
    """
    PCA 降维
    :return:
    """
    data=[[2,8,4,5],[6,3,0,8],[5,4,9,1]]# 3行4列有4个特征
    # 1.实例化一个转换器类
    transfer=PCA(n_components=2)# 如果是小数，则表示保留信息的百分比，如果是整数，则表示保留的特征数，尽量特征减少但信息不损失
    # 2.调用fit_transform
    data_new=transfer.fit_transform(data)
    print(data_new)
    return None
if __name__ == '__main__':
    pca_demo()