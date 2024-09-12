import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1.获取数据
    data=pd.read_csv("D:/factor_returns.csv")
    data=data.iloc[:,1:-2]
    # 2.实例化一个转换器类
    transfer=VarianceThreshold(threshold=5)# 设置阈值，小于5的特征值将被删除
    # 3.调用fit_transform
    data_new=transfer.fit_transform(data)
    print(data_new,data_new.shape)

    # 计算某两个变量之间的相关性
    r1=pearsonr(data["pe_ratio"],data["pb_ratio"])
    print(f"相关系数：{r1}")
    r2 = pearsonr(data["revenue"], data["total_expense"])
    print(f"相关系数：{r2}") 
    return None
if __name__ == '__main__':
    variance_demo()