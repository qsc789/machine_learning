from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
def knn_iris():
    """
    用KNN算法预测鸢尾花种类
    :return:
    """
    # 1.获取数据
    iris=load_iris()
    # 2.划分数据集
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=6)# 四个参数：特征值，目标值，测试集比例，随机种子
    # 3.特征工程：标准化
    transfer=StandardScaler()
    # fit:计算平均值标准值，transform:转换
    x_train=transfer.fit_transform(x_train)# 训练集标准化
    x_test=transfer.transform(x_test)# 测试集标准化，两个数据集操作要相同，要用训练集中的平均值和标准差，所以对测试集不用fit
    # 4.KNN算法预估器
    estimator=KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)
    # 5.模型评估
    # 方法1：直接对比真实值和预测值
    y_pridict=estimator.predict(x_test)
    print(f"预测结果：{y_pridict}")
    print(f"直接对比真实值和预测值：{y_test==y_pridict}")
    # 方法2：计算准确率
    score=estimator.score(x_test,y_test)
    print(f"准确率为：{score}")

    return None


if __name__ == '__main__':
    knn_iris()