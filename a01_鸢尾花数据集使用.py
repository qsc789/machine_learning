from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集,数据集是一个字典
    iris=load_iris()
    print(f"鸢尾花数据集：{iris}")
    print(f"查看数据集描述：{iris["DESCR"]}")
    print(f"查看特征值的名字：{iris.feature_names}")
    print(f"查看特征值：{iris.data}")
    print(f"查看样本数和特征数：{iris.data.shape}")
    # 数据集划分
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.25,random_state=22)
    print(f"训练集特征值：{x_train},{x_train.shape}")
    print(f"验证集特征值：{x_test},{x_test.shape}")
    print(f"训练集目标值：{y_train},{y_train.shape}")
    print(f"验证集目标值：{y_test},{y_test.shape}")


    return None




if __name__ == '__main__':
    # sklearn数据集使用
    datasets_demo()
