import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def minmax_demo():
    """
    归一化
    :return:
    """
    # 1.获取数据
    data=pd.read_csv("D:/dating.txt")
    data=data.iloc[:,:3]# 每行都要前三列

    #　2.实例化一个转换器类
    transfer=MinMaxScaler()# 归一化默认0-1，但设置feature_range=[2,3]也可以调整范围
    # 3.调用fit_transform
    data_new=transfer.fit_transform(data)
    print(f"data_new:{data_new}")
    return None

if __name__ == '__main__':
    minmax_demo()