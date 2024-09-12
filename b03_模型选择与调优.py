from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV# 网格搜索
def knn_iris_gscv():
    """
    用KNN算法对鸢尾花进行分类，添加网格搜索和交叉验证
    :return:
    """
    # 1.获取数据
    iris = load_iris()
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                        random_state=22)  # 四个参数：特征值，目标值，测试集比例，随机种子
    # 3.特征工程：标准化
    transfer = StandardScaler()
    # fit:计算平均值标准值，transform:转换
    x_train = transfer.fit_transform(x_train)  # 训练集标准化
    x_test = transfer.transform(x_test)  # 测试集标准化，两个数据集操作要相同，要用训练集中的平均值和标准差，所以对测试集不用fit
    # 4.KNN算法预估器
    estimator = KNeighborsClassifier()
    # 加入网格搜索和交叉验证
    # 参数准备
    param_dict={"n_neighbors":[1,3,5,7,9,11]}
    estimator=GridSearchCV(estimator,param_grid=param_dict,cv=10)# 三个参数，预估器，参数网格，交叉验证次数
    estimator.fit(x_train, y_train)# 训练模型，提供训练集特征值和训练集目标值

    # 5.模型评估
    # 方法1：直接对比真实值和预测值
    y_pridict = estimator.predict(x_test)
    print(f"预测结果：{y_pridict}")
    print(f"直接对比真实值和预测值：{y_test == y_pridict}")
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print(f"准确率为：{score}")
    print(f"最佳参数：{estimator.best_params_}")
    print(f"最佳结果：{estimator.best_score_}")
    print(f"最佳估计器：{estimator.best_estimator_}")
    print(f"交叉验证结果：{estimator.cv_results_}")

    return None


if __name__ == '__main__':
   knn_iris_gscv()
