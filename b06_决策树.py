from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
def decision_iris():
    """
    用决策树分类鸢尾花
    :return:
    """
    iris=load_iris()
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=22)
    estimator=DecisionTreeClassifier(criterion='entropy')# criterion='entropy'表示信息熵，越大越好
    estimator.fit(x_train,y_train)
    y_predict=estimator.predict(x_test)
    print(y_test==y_predict)

    score=estimator.score(x_test,y_test)
    print(score)
    # 导出决策树
    export_graphviz(estimator,out_file='tree.dot')

if __name__ == '__main__':
    decision_iris()

